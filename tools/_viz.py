"""绘图公用件：后端选择、字体、以及"打击泳道"。

为什么单独抽出来
--------------
`dump_analysis.py` 和 `show_envelopes.py` 都需要：
  1) 在无 GUI 环境下安全地选后端（否则 matplotlib 静默退回 agg，窗口不弹）
  2) 把合并后的 Hit 画成**顶部短竖线泳道**，而不是贯穿整个波形面板的长线
     （长线在 1800+ 次打击时会把波形完全盖住，什么都看不见）
"""

from __future__ import annotations

import numpy as np

from onset_detect.hits import CLICK, LOW, Hit

NON_GUI_BACKENDS = frozenset({"agg", "pdf", "svg", "ps", "cairo", "template"})

# 频带类别 -> (显示名, 颜色)。顺序与 hit_class() 的返回值对应。
CLASS_STYLE = (
    ("低频(LOW)", "#e6194b"),
    ("高频(CLICK)", "#42d4f4"),
    ("低频+高频", "#f58231"),
    ("其它(贝斯/混合)", "#bf9000"),
)


def pick_backend(show: bool) -> bool:
    """选择 matplotlib 后端；返回最终是否具备可弹窗的 GUI 后端。

    matplotlib 找不到 GUI 工具包时会**静默退回 agg**，此时 plt.show() 不弹窗、
    只打印一行警告。所以这里主动按 Qt -> Tk -> wx 的顺序尝试，并给出修复提示。
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
        return False
    if matplotlib.get_backend().lower() not in NON_GUI_BACKENDS:
        return True
    for name in ("QtAgg", "TkAgg", "WXAgg"):
        try:
            matplotlib.use(name, force=True)
            print(f"[matplotlib] 使用交互后端: {name}")
            return True
        except Exception:
            continue
    print("[matplotlib] 没有可用的 GUI 后端，无法弹窗。")
    print("             修复: python -m pip install PyQt5")
    return False


def setup_fonts() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def hit_class(bands) -> int:
    """把一次打击归到 4 个粗类别之一（只用于显示配色）。"""
    s = set(bands)
    low, click = bool(s & LOW), bool(s & CLICK)
    if low and click:
        return 2
    if low:
        return 0
    if click:
        return 1
    return 3


def draw_hit_lane(ax, hits: list[Hit], y0: float = 1.05, y1: float = 1.20, lw: float = 0.7):
    """在波形面板顶部画"打击泳道"。

    每个打击画一根从 y0 到 y1 的短竖线，颜色表示它的频带类别。
    波形本身的幅度在 -1..1 之间，所以泳道放在 1.05~1.20 不会互相遮挡。
    """
    if not hits:
        return
    times = np.array([h.t for h in hits], dtype=float)
    cls = np.array([hit_class(h.bands) for h in hits], dtype=int)
    for k, (name, color) in enumerate(CLASS_STYLE):
        m = cls == k
        if m.any():
            ax.vlines(
                times[m], y0, y1, color=color, lw=lw, alpha=0.9, label=f"{name} ({int(m.sum())})"
            )
    ax.axhline(y0, color="#888888", lw=0.6, alpha=0.8)


def hit_time_index(hits: list[Hit]) -> np.ndarray:
    """预取所有打击时刻，便于用 searchsorted 快速找"最近的打击"。"""
    return np.array([h.t for h in hits], dtype=float)


def nearest_hit(hits: list[Hit], times: np.ndarray, t: float, tol: float = 0.030):
    """返回 t 附近 ±tol 内最近的打击（没有则 None）。用于鼠标读数/单击。"""
    if times.size == 0:
        return None
    i = int(np.searchsorted(times, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < times.size:
            d = abs(float(times[j]) - t)
            if d <= tol and (best is None or d < best[0]):
                best = (d, hits[j])
    return None if best is None else best[1]
