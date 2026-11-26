"""把 10 条带 flux 变成"几条乐器强度包络"，并做节奏自检。

回答的问题
==========
    "手里这堆带 flux，怎么变成节拍器能吃的输入？"
    答: 合成 3~4 条**乐器强度包络**（pulse / snare / hat），
        并顺便用自相关估一个 BPM，验证这条曲线里真的含节拍信息。

用法:
    python tools/show_envelopes.py "test_case_日不落-蔡依林.wav"
    python tools/show_envelopes.py "..." --zoom 12 24 --out artifacts/env.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onset_detect.analysis import analyze_file  # noqa: E402
from onset_detect.bands import BAND_NAMES, N_BANDS  # noqa: E402
from onset_detect.envelope import estimate_tempo_bpm, instrument_envelopes  # noqa: E402
from onset_detect.fxp import q88_to_db  # noqa: E402
from onset_detect.hits import CLICK, LOW, group_hits  # noqa: E402
from onset_detect.params import Params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _viz import draw_hit_lane, setup_fonts  # noqa: E402

_NON_GUI = {"agg", "pdf", "svg", "ps", "cairo", "template"}


def _pick_backend(show: bool) -> bool:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
        return False
    if matplotlib.get_backend().lower() not in _NON_GUI:
        return True
    for name in ("QtAgg", "TkAgg", "WXAgg"):
        try:
            matplotlib.use(name, force=True)
            print(f"[matplotlib] 使用交互后端: {name}")
            return True
        except Exception:
            continue
    print("[matplotlib] 无 GUI 后端; 修复: python -m pip install PyQt5")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", dest="show", action="store_true", default=True)
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--zoom", nargs=2, type=float, default=None, metavar=("T0", "T1"))
    ap.add_argument("--float", action="store_true")
    args = ap.parse_args()

    gui_ok = _pick_backend(args.show)
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    setup_fonts()

    p = Params(fixed=not args.float)
    events, tr, x, sr = analyze_file(args.wav, p, block=256)
    assert tr is not None
    dur = len(x) / sr
    fps = sr / p.hop

    hits = group_hits(events, window_s=0.040)
    envs = instrument_envelopes(tr["band_flux"])
    ft = tr["frame_center"] / sr

    print(f"\n文件: {args.wav}  {dur:.1f} s")
    print(f"打击数: {len(hits)}  ({len(hits) / dur:.2f} /s)")
    for name, e in envs.items():
        bpm, strength = estimate_tempo_bpm(e, fps)
        print(f"  {name:10s} 自相关 BPM = {bpm:6.1f}   周期性强度 = {strength:.2f}")

    # 按"粗略类别"统计（只是给人看的，P3 换成真分类器）
    cls = {"低频(LOW)": 0, "高频(CLICK)": 0, "低频+高频": 0, "仅贝斯": 0, "其它": 0}
    for h in hits:
        s = set(h.bands)
        low, click = bool(s & LOW), bool(s & CLICK)
        if low and click:
            cls["低频+高频"] += 1
        elif low:
            cls["低频(LOW)"] += 1
        elif click:
            cls["高频(CLICK)"] += 1
        elif s:
            cls["仅贝斯"] += 1
        else:
            cls["其它"] += 1
    print("打击粗分类:", ", ".join(f"{k}={v}" for k, v in cls.items()))

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True, constrained_layout=True)

    # ---- 1. 波形 + 打击泳道（顶部短竖线，颜色=频带类别）----
    step = max(1, len(x) // 8000)
    xb = x[: (len(x) // step) * step].reshape(-1, step)
    tt = np.arange(xb.shape[0]) * step / sr
    axes[0].fill_between(tt, xb.min(1) / 32768.0, xb.max(1) / 32768.0, color="#333333", lw=0)
    axes[0].set_ylabel("waveform")
    axes[0].set_ylim(-1.05, 1.24)
    draw_hit_lane(axes[0], hits, y0=1.05, y1=1.20)
    axes[0].legend(loc="lower right", ncol=4, fontsize=7, framealpha=0.85)
    axes[0].set_title("波形 + 合并后的打击（顶部泳道, 颜色=频带类别）", fontsize=9)

    # ---- 2. 三条乐器强度包络（这就是"能喂给节拍器"的东西）----
    axes[1].plot(ft, q88_to_db(envs["pulse"]), lw=0.7, color="#e6194b", label="pulse (sub/kick)")
    axes[1].plot(ft, q88_to_db(envs["snare"]), lw=0.7, color="#f58231", label="snare (presence/treble)")
    axes[1].plot(ft, q88_to_db(envs["hat"]), lw=0.7, color="#42d4f4", label="hat (high/air)")
    axes[1].set_ylabel("onset strength (dB)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=7)
    axes[1].set_title("乐器强度包络 —— 下游节拍跟踪的直接输入", fontsize=9)
    axes[1].grid(alpha=0.2)

    # ---- 3. bass 抑制的代价：pulse vs pulse_net ----
    axes[2].plot(ft, q88_to_db(envs["pulse"]), lw=0.7, color="#e6194b", label="pulse（原始）")
    axes[2].plot(ft, q88_to_db(envs["pulse_net"]), lw=0.7, color="#3cb44b", label="pulse_net（减掉半个 bass 带）")
    axes[2].set_ylabel("dB")
    axes[2].legend(loc="upper right", ncol=2, fontsize=7)
    axes[2].set_title("廉价『减 bass』的取舍：去掉贝斯音符的同时也会压掉被掩蔽的真底鼓（P3 用 HPSS 替换）", fontsize=9)
    axes[2].grid(alpha=0.2)

    # ---- 4. 指纹散点：x=时间, y=带, 点大小=强度 ----
    hx, hy = [], []
    for h in hits:
        for b, s in zip(h.bands, h.strength_q88):
            hx.append(h.t)
            hy.append(b)
    hx = np.asarray(hx)
    hy = np.asarray(hy)
    axes[3].scatter(hx, hy, s=6, c="#333333", alpha=0.5, linewidths=0)
    axes[3].set_yticks(range(N_BANDS))
    axes[3].set_yticklabels(BAND_NAMES, fontsize=7)
    axes[3].set_ylabel("band")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylim(-0.6, N_BANDS - 0.4)
    axes[3].set_title("打击在 10 个带上的分布（每次打击的『指纹』）", fontsize=9)
    axes[3].grid(alpha=0.2)

    # ---- 初始视窗 ----
    if args.zoom:
        z0, z1 = max(0.0, args.zoom[0]), min(dur, args.zoom[1])
        if z1 > z0:
            axes[-1].set_xlim(z0, z1)

    # ---- 鼠标读数 ----
    vlines = [ax.axvline(float(ft[0]), color="#00e5ff", lw=0.9, alpha=0.0) for ax in axes]
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
    nf = envs["pulse"].shape[0]
    state = {"txt": ""}

    def _on_move(ev):
        if ev.inaxes is None or ev.xdata is None:
            if state["txt"]:
                for ln in vlines:
                    ln.set_alpha(0.0)
                readout.set_text("")
                state["txt"] = ""
                fig.canvas.draw_idle()
            return
        k = int(np.clip(round((float(ev.xdata) * sr - p.win // 2) / p.hop), 0, nf - 1))
        txt = (
            f"t={k / fps:8.3f}s  pulse={q88_to_db(envs['pulse'][k]):5.1f}  "
            f"snare={q88_to_db(envs['snare'][k]):5.1f}  hat={q88_to_db(envs['hat'][k]):5.1f} dB"
        )
        if txt == state["txt"]:
            return
        state["txt"] = txt
        for ln in vlines:
            ln.set_xdata([k / fps, k / fps])
            ln.set_alpha(0.9)
        readout.set_text(txt)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.suptitle(f"{Path(args.wav).name}  hits={len(hits)}  {len(hits) / dur:.2f}/s", fontsize=10)

    out = args.out or (None if gui_ok else "artifacts/envelopes.png")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(f"已写出 {out}")
    if args.show and gui_ok:
        print("交互: 左键拖动缩放 / 右键拖动平移 / 鼠标移动读数 / Home 恢复")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
