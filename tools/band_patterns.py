"""把"每带各自出事件"聚合成"打击 + 频带指纹"，检查数据到底能不能用。

为什么需要这一步
==============
P0 的输出是 10 个带**各自独立**的事件（实测 228 s 出 5414 个 = 23.7/s）。
同一次物理打击会同时在多个带触发，所以直接拿这些事件去驱动节拍器，
既过量又重复 —— 这就是你在图上看到"每条线都亮"的原因。

本工具把时间上相近的事件合并成一个 **hit（打击）**，再看它的 10 维
**频带指纹**（哪些带同时亮了、各多强），并统计指纹的种类分布：

    如果几百种指纹里少数几种占绝大多数  ->  数据是"可分"的
    ->  后面就能用简单规则/小分类器把 底鼓 / 军鼓 / hi-hat / 贝斯 分出来

用法:
    python tools/band_patterns.py "test_case_日不落-蔡依林.wav"
    python tools/band_patterns.py "..." --top 20 --window-ms 40 --min-count 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onset_detect.analysis import analyze_file  # noqa: E402
from onset_detect.bands import (  # noqa: E402
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
from onset_detect.fxp import q88_to_db  # noqa: E402
from onset_detect.params import Params  # noqa: E402

LOW = {B_SUB, B_KICK}
CLICK = {B_PRESENCE, B_TREBLE, B_HIGH, B_AIR}
BASSY = {B_BASS, B_LOWMID}


def _label(bands: set[int]) -> str:
    """粗粒度猜测（P3 会用真正的特征+分类器替换它，这里只是给人看的标签）。"""
    n = len(bands)
    low, click, bassy = bool(bands & LOW), bool(bands & CLICK), bool(bands & BASSY)
    if n >= 6:
        return "宽带重击(军鼓/拍手/段落过渡)"
    if low and click and not bassy:
        return "底鼓(带 click)"
    if low and not click and not bassy:
        return "底鼓/808(无 click)"
    if click and not low and not bassy:
        return "hi-hat / 镲"
    if low and bassy and click:
        return "鼓+贝斯同时(强拍)"
    if low and bassy:
        return "底鼓+贝斯(掩蔽场景)"
    if bassy and click:
        return "贝斯+镲 或 军鼓"
    if bassy:
        return "贝斯音符"
    return "混合/待定"


def _pattern_str(bands: set[int]) -> str:
    return "".join("#" if b in bands else "." for b in range(N_BANDS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--window-ms", type=float, default=40.0, help="合并成一次打击的时间窗")
    ap.add_argument("--top", type=int, default=15, help="列出前 N 种指纹")
    ap.add_argument("--min-count", type=int, default=5, help="只显示出现次数 >= 该值的指纹")
    ap.add_argument("--float", action="store_true")
    args = ap.parse_args()

    p = Params(fixed=not args.float)
    events, tr, x, sr = analyze_file(args.wav, p, block=256)
    assert tr is not None
    dur = len(x) / sr
    flux_db = q88_to_db(tr["band_flux"])

    # ---- 1. 把事件按时间聚成 hit ----
    events = sorted(events, key=lambda e: e.t)
    hits: list[list] = []
    cur: list = []
    for e in events:
        if not cur or e.t - cur[0].t <= args.window_ms * 1e-3:
            cur.append(e)
        else:
            hits.append(cur)
            cur = [e]
    if cur:
        hits.append(cur)

    # ---- 2. 每个 hit 的频带指纹 ----
    patterns: Counter[tuple[int, ...]] = Counter()
    strength: dict[tuple[int, ...], np.ndarray] = {}
    for h in hits:
        key = tuple(sorted({e.band for e in h}))
        patterns[key] += 1
        if key not in strength:
            strength[key] = np.zeros(N_BANDS, dtype=np.float64)
        for e in h:
            strength[key][e.band] = max(strength[key][e.band], q88_to_db(e.strength_q88))

    nb_per_hit = np.array([len(h) for h in hits])
    print(f"\n文件: {args.wav}  {dur:.1f} s   fixed={p.fixed}")
    print(f"原始事件: {len(events)}  ({len(events) / dur:.2f} /s)")
    print(f"合并后 hit: {len(hits)}  ({len(hits) / dur:.2f} /s)   ← 这才是『打击』的真实密度")
    print(f"每次 hit 平均占用带数: {nb_per_hit.mean():.2f}   (1 个带 = 无法判别的孤立事件)")
    print(f"  只有 1 个带亮的 hit: {int((nb_per_hit == 1).sum())} 次 "
          f"({100 * (nb_per_hit == 1).mean():.1f}%)")
    print(f"  同时亮 >=6 个带的 hit: {int((nb_per_hit >= 6).sum())} 次 "
          f"({100 * (nb_per_hit >= 6).mean():.1f}%)")

    # ---- 3. 指纹分布 ----
    print(f"\n频带指纹  (列顺序: {' '.join(n[:3] for n in BAND_NAMES)})")
    print(f"{'次数':>6} {'占比':>6}  指纹                 猜测")
    print("-" * 78)
    shown = 0
    for key, cnt in patterns.most_common():
        if cnt < args.min_count:
            continue
        if shown >= args.top:
            break
        shown += 1
        bands = set(key)
        print(f"{cnt:6d} {100 * cnt / len(hits):5.1f}%  {_pattern_str(bands)}  {_label(bands)}")
    covered = sum(c for k, c in patterns.items() if c >= args.min_count)
    print("-" * 78)
    print(f"共 {len(patterns)} 种指纹; 出现 >= {args.min_count} 次的 "
          f"{sum(1 for c in patterns.values() if c >= args.min_count)} 种，"
          f"覆盖 {100 * covered / len(hits):.1f}% 的 hit")

    # ---- 4. 共现矩阵：哪些带总是一起亮 ----
    print("\n带间共现率 (%)  —— 行/列 同时出现在同一个 hit 里的比例")
    print("        " + "".join(f"{n[:4]:>6}" for n in BAND_NAMES))
    co = np.zeros((N_BANDS, N_BANDS))
    for key in patterns:
        for a in key:
            for b in key:
                co[a, b] += 1
    co = 100.0 * co / max(1, len(hits))
    for a in range(N_BANDS):
        print(f"{BAND_NAMES[a]:7s} " + "".join(f"{co[a, b]:6.1f}" for b in range(N_BANDS)))

    # ---- 5. 每带单独触发（噪声带诊断）----
    print("\n每带参与 hit 的次数 / 独自一人的次数")
    alone = Counter()
    part = Counter()
    for key, cnt in patterns.items():
        for b in key:
            part[b] += cnt
        if len(key) == 1:
            alone[key[0]] += cnt
    for b in range(N_BANDS):
        print(f"  {BAND_NAMES[b]:9s} 参与 {part[b]:5d}  单独 {alone[b]:5d} "
              f"({100 * alone[b] / max(1, part[b]):5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
