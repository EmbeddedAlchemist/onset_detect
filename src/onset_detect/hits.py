"""把"每带各自的事件"合并成"打击（hit）"—— 这是喂给判别器的单位。

为什么必须合并
============
一次物理打击会**同时**在多个带触发：
    底鼓 ≈ sub + kick +（click 带 presence/treble）
    hi-hat ≈ high + air
    ？？？ = bass 单独 / sub 单独  ← 这些才是真正需要判别的地方

P0 每个带各自出事件，直接把 5414 个事件当"鼓点"会让节拍器被同一击重复驱动。
合并后得到"一次打击 + 10 维频带指纹"，它才是后续判别（乐器分类）的输入。

流式实现
--------
合并需要往后等 30~50 ms（等同一击在其它带的事件到齐），所以是个**延迟队列**：
新事件进来时，把已经超过窗口、不会再变的组吐出去。
只延迟事件，不窗口化 PCM —— 内存固定，且不算前瞻。
"""

from __future__ import annotations

from dataclasses import dataclass

from .bands import (
    B_AIR,
    B_BASS,
    B_HIGH,
    B_KICK,
    B_LOWMID,
    B_PRESENCE,
    B_SUB,
    B_TREBLE,
    BAND_NAMES,
    N_BANDS,
)
from .detector import Event

LOW = frozenset({B_SUB, B_KICK})
CLICK = frozenset({B_PRESENCE, B_TREBLE, B_HIGH, B_AIR})
BASSY = frozenset({B_BASS, B_LOWMID})


def hit_label(bands) -> str:
    """粗粒度猜测标签（P3 会用真正的特征+分类器替换，这里只供人工查看）。"""
    b = set(bands)
    n = len(b)
    low, click, bassy = bool(b & LOW), bool(b & CLICK), bool(b & BASSY)
    if n >= 6:
        return "宽带重击(军鼓/拍手/过渡)"
    if low and click and not bassy:
        return "底鼓(带click)"
    if low and not click and not bassy:
        return "底鼓/808(无click)"
    if click and not low and not bassy:
        return "hi-hat/镲"
    if low and bassy and click:
        return "鼓+贝斯(强拍)"
    if low and bassy:
        return "底鼓+贝斯(掩蔽场景)"
    if bassy and click:
        return "贝斯+镲 或 军鼓"
    if bassy:
        return "贝斯音符"
    return "混合/待定"


@dataclass(frozen=True)
class Hit:
    """一次物理打击（由若干带的起振事件合并而成）。

    bands / strength_q88 按同一顺序对齐，都是触发该打击的带。
    """

    t: float
    sample: int
    bands: tuple[int, ...]
    strength_q88: tuple[int, ...]

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def pattern(self) -> str:
        """10 位指纹，'#' 表示该带在这一击里亮了（顺序同 BAND_NAMES）。"""
        s = set(self.bands)
        return "".join("#" if b in s else "." for b in range(N_BANDS))

    @property
    def band_names(self) -> str:
        return "+".join(BAND_NAMES[b] for b in self.bands)

    @property
    def label(self) -> str:
        return hit_label(self.bands)


def _to_hit(group: list[Event], window_s: float) -> Hit:
    best: dict[int, int] = {}
    for e in group:
        if e.band not in best or e.strength_q88 > best[e.band]:
            best[e.band] = e.strength_q88
    bands = tuple(sorted(best))
    return Hit(
        t=group[0].t,
        sample=group[0].sample,
        bands=bands,
        strength_q88=tuple(best[b] for b in bands),
    )


def group_hits(events, window_s: float = 0.040) -> list[Hit]:
    """批量合并（离线用；流式请用 HitMerger）。"""
    evs = sorted(events, key=lambda e: e.t)
    out: list[Hit] = []
    cur: list[Event] = []
    for e in evs:
        if not cur or e.t - cur[0].t <= window_s:
            cur.append(e)
        else:
            out.append(_to_hit(cur, window_s))
            cur = [e]
    if cur:
        out.append(_to_hit(cur, window_s))
    return out


class HitMerger:
    """流式合并器：喂入事件 + "当前时刻"，吐出已定型的打击。

    用法（C 侧同样结构，只是一个定长环形队列）:
        m = HitMerger(window_s=0.040)
        for ev in detector.process_block(x):
            yield from m.push(ev, now_s=...)
    """

    __slots__ = ("window_s", "_cur", "_last_t")

    def __init__(self, window_s: float = 0.040):
        self.window_s = float(window_s)
        self._cur: list[Event] = []
        self._last_t = 0.0

    def push(self, events, now_s: float) -> list[Hit]:
        out: list[Hit] = []
        self._last_t = now_s
        for e in events:
            if self._cur and e.t - self._cur[0].t > self.window_s:
                out.append(_to_hit(self._cur, self.window_s))
                self._cur = []
            self._cur.append(e)
        # 当前组已经超过窗口 => 不会再变化，可以输出
        if self._cur and now_s - self._cur[0].t > self.window_s:
            out.append(_to_hit(self._cur, self.window_s))
            self._cur = []
        return out

    def flush(self) -> list[Hit]:
        out = [_to_hit(self._cur, self.window_s)] if self._cur else []
        self._cur = []
        return out
