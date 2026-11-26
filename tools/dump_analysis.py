"""整曲分析 + 交互式查看（波形 / 10 带电平 / 带 flux / 检出点 / 统计）。

用法:
    # 默认：弹出可交互窗口
    python tools/dump_analysis.py "test_case_日不落-蔡依林.wav"

    # 只看 12~24 秒（之后仍可在窗口里缩放/平移）
    python tools/dump_analysis.py "..." --zoom 12 24

    # 同时导出 PNG（窗口仍会弹出）
    python tools/dump_analysis.py "..." --out artifacts/dump.png

    # 不弹窗，只导出 PNG（无显示环境 / 批处理）
    python tools/dump_analysis.py "..." --out artifacts/dump.png --no-show

窗口里的操作:
    鼠标左键拖动 / 滚轮 / 工具栏放大镜  -> 缩放
    鼠标右键拖动                        -> 平移
    鼠标移动                            -> 顶部显示时间、帧号、该帧最强的带
    左键单击                            -> 在控制台打印附近 ±30 ms 的检出事件
    工具栏 Home 图标 / 键盘 h           -> 恢复全视图

注意: 这里走的是与设备完全相同的流式入口（逐块喂入），不是离线批处理。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onset_detect.analysis import analyze_file  # noqa: E402
from onset_detect.bands import BAND_NAMES  # noqa: E402
from onset_detect.fxp import q88_to_db  # noqa: E402
from onset_detect.hits import group_hits  # noqa: E402
from onset_detect.params import Params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _viz import draw_hit_lane, hit_time_index, nearest_hit, setup_fonts  # noqa: E402


BAND_COLORS = (
    "#e6194b",
    "#f58231",
    "#bf9000",
    "#8f9e00",
    "#3cb44b",
    "#42d4f4",
    "#4363d8",
    "#911eb4",
    "#f032e6",
    "#808080",
)


def _summary(events, sr: int) -> str:
    n = len(events)
    lines = [f"总事件数: {n}  ({n / (max(1, events[-1].t) if events else 1):.2f} /s)"]
    counts = np.zeros(len(BAND_NAMES), dtype=int)
    for e in events:
        counts[e.band] += 1
    for i, c in enumerate(counts):
        lines.append(f"  {i} {BAND_NAMES[i]:9s} {c:5d}")
    # 按 30 ms 聚类，得到"物理打击"数量
    grouped = []
    last = -1e9
    for e in events:
        if e.t - last > 0.030:
            grouped.append(e.t)
        last = e.t
    lines.append(f"聚类后(30 ms)打击数: {len(grouped)}")
    return "\n".join(lines)


def _envelope(x: np.ndarray, max_bins: int = 6000):
    """把长波形压成 min/max 包络（只影响显示，不影响分析）。

    为什么要压: 228 s 有 366 万个采样点，全画出来会让交互缩放很卡。
    每 max_bins 个点取一次 min/max，既保留"看起来最响的地方"，点数又少了两个数量级。

    返回 (idx, lo, hi)：idx 是每个小段的起始采样下标。
    """
    n = x.shape[0]
    if n <= 2 * max_bins:
        idx = np.arange(n)
        return idx, x, x
    step = int(np.ceil(n / max_bins))
    m = (n // step) * step
    xb = x[:m].reshape(-1, step)
    return np.arange(0, m, step), xb.min(axis=1), xb.max(axis=1)


_NON_GUI_BACKENDS = {"agg", "pdf", "svg", "ps", "cairo", "template"}


def _pick_backend(show: bool) -> bool:
    """兼容保留：实际实现见 tools/_viz.py（两个工具共用同一份逻辑）。"""
    from _viz import pick_backend

    return pick_backend(show)


def main() -> int:
    ap = argparse.ArgumentParser(description="整曲流式分析 + 交互式可视化")
    ap.add_argument("wav")
    ap.add_argument("--out", default=None, help="可选: 同时导出 PNG 的路径")
    ap.add_argument(
        "--show", dest="show", action="store_true", default=True, help="弹出交互窗口（默认）"
    )
    ap.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="不弹窗，仅导出 PNG（无显示环境 / 批处理）",
    )
    ap.add_argument("--zoom", nargs=2, type=float, default=None, metavar=("T0", "T1"))
    ap.add_argument("--float", action="store_true", help="用浮点研究模式")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument(
        "--per-band",
        dest="per_band",
        action="store_true",
        default=False,
        help="同时画出每个带各自的原始事件（默认只画合并后的打击）",
    )
    ap.add_argument("--hit-window-ms", type=float, default=40.0, help="合并成一次打击的时间窗")
    args = ap.parse_args()

    # 先决定后端、再 import pyplot（pyplot 一旦导入就锁定后端）
    import matplotlib

    gui_ok = _pick_backend(args.show)
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    setup_fonts()

    p = Params(fixed=not args.float)
    t0 = time.perf_counter()
    events, tr, x, sr = analyze_file(args.wav, p, block=args.block)
    dt = time.perf_counter() - t0
    dur = len(x) / sr
    print(f"文件: {args.wav}  {dur:.1f} s  {sr} Hz")
    print(f"耗时: {dt:.2f} s  =>  {dur / max(dt, 1e-9):.1f}x realtime  (fixed={p.fixed})")
    print(_summary(events, sr))
    assert tr is not None


    ft = tr["frame_center"] / sr
    band_L_db = q88_to_db(tr["band_L"] - int(tr["band_L"].max()))
    flux_db = q88_to_db(tr["band_flux"])
    thr_db = q88_to_db(tr["band_thr"])

    # 全曲都画进图里（只画一次），--zoom 只决定"初始视窗"，
    # 之后仍可在窗口里随意平移/缩放查看整首歌。
    sel = slice(None)
    wsel = slice(None)
    t_lo = float(ft[0])

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True, constrained_layout=True)

    # ---- 面板 1: 波形 + 合并后的"打击泳道" ----
    # y > 1.05 的区域专门留给打击标记，不再用贯穿全高的竖线（会盖住波形）
    idx, lo, hi = _envelope(x[wsel], max_bins=6000)
    tt = (wsel.start or 0) / sr + idx / sr
    axes[0].fill_between(tt, lo / 32768.0, hi / 32768.0, color="#333333", lw=0)
    axes[0].set_ylabel("waveform")
    axes[0].set_ylim(-1.05, 1.24)

    hits = group_hits(events, window_s=args.hit_window_ms * 1e-3)
    draw_hit_lane(axes[0], hits, y0=1.05, y1=1.20, lw=3.0)
    hit_times = hit_time_index(hits)

    if args.per_band:  # 需要看每个带的原始事件时才画（很密，默认关闭）
        for b in range(len(BAND_NAMES)):
            et = np.array([e.t for e in events if e.band == b])
            if et.size:
                axes[0].vlines(et, -1.0, 1.0, color=BAND_COLORS[b], lw=0.5, alpha=0.30)

    axes[0].legend(loc="lower right", ncol=4, fontsize=7, framealpha=0.85)

    # ---- 面板 2: 10 个带的电平（dB，相对全曲最大值）----
    ext = (ft[sel][0], ft[sel][-1], -0.5, len(BAND_NAMES) - 0.5)
    im1 = axes[1].imshow(
        band_L_db[sel, :].T,
        aspect="auto",
        origin="lower",
        extent=ext,
        cmap="magma",
        vmin=-60,
        vmax=0,
        interpolation="nearest",
    )
    axes[1].set_yticks(range(len(BAND_NAMES)))
    axes[1].set_yticklabels(BAND_NAMES, fontsize=7)
    axes[1].set_ylabel("band level (dB re max)")
    fig.colorbar(im1, ax=axes[1], pad=0.005)

    # ---- 面板 3: 谱通量（= 起振强度），亮的地方就是"能量在涨"----
    im2 = axes[2].imshow(
        flux_db[sel, :].T,
        aspect="auto",
        origin="lower",
        extent=ext,
        cmap="viridis",
        vmin=0,
        vmax=12,
        interpolation="nearest",
    )
    axes[2].set_yticks(range(len(BAND_NAMES)))
    axes[2].set_yticklabels(BAND_NAMES, fontsize=7)
    axes[2].set_ylabel("band flux (dB)")
    fig.colorbar(im2, ax=axes[2], pad=0.005)

    # ---- 面板 4: 低 4 个带的 flux 曲线 + 阈值（虚线）----
    for b in (0, 1, 2, 3):
        axes[3].plot(
            ft[sel], flux_db[sel, b], lw=0.7, color=BAND_COLORS[b], label=f"{BAND_NAMES[b]} flux"
        )
        axes[3].plot(ft[sel], thr_db[sel, b], lw=0.6, ls="--", color=BAND_COLORS[b], alpha=0.6)
    axes[3].set_ylabel("flux / thr (dB)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", ncol=4, fontsize=7)
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        f"{Path(args.wav).name}   dur={dur:.1f}s   fixed={p.fixed}   "
        f"事件={len(events)}  打击={len(hits)}  {dur / max(dt, 1e-9):.0f}x RT",
        fontsize=10,
    )
    try:  # 交互后端才有窗口标题
        fig.canvas.manager.set_window_title(f"onset_detect — {Path(args.wav).name}")
    except Exception:
        pass

    # ---- 鼠标交互：十字线 + 读数 + 单击打印附近事件 ----
    # 注意: 初值不能设成 0，否则这条不可见的线会把坐标轴自动缩放撑到 0 秒
    vlines = [ax.axvline(t_lo, color="#00e5ff", lw=0.9, alpha=0.0) for ax in axes]
    readout = axes[0].text(
        0.005,
        0.955,
        "",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="w",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55),
    )
    nf = flux_db.shape[0]
    state = {"text": ""}

    def _on_move(ev):
        if ev.inaxes is None or ev.xdata is None:
            if state["text"]:
                for ln in vlines:
                    ln.set_alpha(0.0)
                readout.set_text("")
                state["text"] = ""
                fig.canvas.draw_idle()
            return
        t = float(ev.xdata)
        k = int(round((t * sr - p.win // 2) / p.hop))
        k = max(0, min(nf - 1, k))
        row = flux_db[k]
        b = int(np.argmax(row))
        h = nearest_hit(hits, hit_times, t)
        txt = (
            f"t={t:8.3f}s  frame={k:5d}  "
            f"最强带: {BAND_NAMES[b]:8s} {row[b]:5.1f} dB (thr {thr_db[k, b]:4.1f})"
        )
        if h is not None:
            txt += f"  |  打击 {h.pattern} {h.label}"
        if txt == state["text"]:
            return
        state["text"] = txt
        for ln in vlines:
            ln.set_xdata([t, t])
            ln.set_alpha(0.9)
        readout.set_text(txt)
        fig.canvas.draw_idle()

    def _on_click(ev):
        if ev.inaxes is None or ev.xdata is None or ev.button != 1:
            return
        t = float(ev.xdata)
        near = [h for h in hits if abs(h.t - t) <= 0.030]
        if not near:
            print(f"[{t:8.3f}s] 附近 ±30 ms 无打击")
        for h in near:
            bands = ", ".join(
                f"{BAND_NAMES[b]}:{q88_to_db(s):.1f}dB" for b, s in zip(h.bands, h.strength_q88)
            )
            print(
                f"[{h.t:8.3f}s] {h.pattern}  {h.label:22s} "
                f"({h.n_bands} 带, Δ={1000.0 * (h.t - t):+5.0f} ms)  {bands}"
            )

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    # ---- 初始视窗（--zoom）----
    if args.zoom:
        z0 = max(0.0, args.zoom[0])
        z1 = min(dur, args.zoom[1])
        if z1 > z0:
            axes[-1].set_xlim(z0, z1)  # sharex: 设一个就是全部
            print(f"初始视窗: {z0:.2f} ~ {z1:.2f} s（可拖动平移查看全曲）")

    # ---- 导出 / 显示 ----
    # 没窗口可用时也要保证用户能拿到结果，所以自动补一个 PNG 导出
    out = args.out
    if out is None and (not args.show or not gui_ok):
        out = "artifacts/dump.png"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(f"已写出 {out}")

    if args.show and gui_ok:
        print(
            "交互提示: 左键拖动=缩放, 右键拖动=平移, 鼠标移动=读数, "
            "单击=打印附近事件, Home 键=恢复全视图"
        )
        plt.show()
    elif args.show:
        print("（当前环境没有 GUI 后端，未弹窗，已改为导出 PNG）")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
