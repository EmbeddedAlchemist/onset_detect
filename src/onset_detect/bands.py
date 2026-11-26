"""10 个对数子带的 bin 边界（与 target C 完全一致的常量表）。

这一步在做什么
--------------
把 FFT 输出的 513 个频点"分桶"成 10 个频带，每个带算一个总能量。

为什么必须分带（而不是 0-300 Hz 全加起来）
----------------------------------------
贝斯的谐波占了低频里大部分 bin 的能量，全加起来时总和≈贝斯，
底鼓只占少数 bin，贡献被"稀释" -> 叠在贝斯上的底鼓检不出来。
分成 10 个带之后，"贝斯主要落在 bass 带、底鼓主要落在 sub/kick 带"，
才有可能分别判断。

为什么取这样的边界
----------------
近似对数间距（每个带比上一个带宽约 1.8 倍），这样每个带里"听起来"的
频率内容是相近的，而且能覆盖 20 Hz ~ 8 kHz 全范围。

    16 kHz / 1024 点 => 15.625 Hz/bin
    b0 sub       [1,  4)     15.6 -  62.5 Hz   底鼓 thump / sub
    b1 kick      [4,  7)     62.5 - 109   Hz   底鼓 body
    b2 bass      [7, 13)    109   - 203   Hz   贝斯 f0
    b3 lowmid    [13,26)    203   - 406   Hz   贝斯 H2/H3、军鼓体
    b4 mid       [26,51)    406   - 797   Hz
    b5 uppermid  [51,96)    797   - 1500  Hz
    b6 presence  [96,160)   1500  - 2500  Hz   军鼓噪声、click
    b7 treble    [160,256)  2500  - 4000  Hz   click 主体
    b8 high      [256,352)  4000  - 5500  Hz   hi-hat
    b9 air       [352,513]  5500  - 8000  Hz   hi-hat / cymbal
"""


from __future__ import annotations

import numpy as np

BAND_NAMES = (
    "sub",
    "kick",
    "bass",
    "lowmid",
    "mid",
    "uppermid",
    "presence",
    "treble",
    "high",
    "air",
)

BAND_EDGES_HZ = (
    (20.0, 63.0),
    (63.0, 110.0),
    (110.0, 203.0),
    (203.0, 406.0),
    (406.0, 797.0),
    (797.0, 1500.0),
    (1500.0, 2500.0),
    (2500.0, 4000.0),
    (4000.0, 5500.0),
    (5500.0, 8000.0),
)

N_BANDS = len(BAND_NAMES)

# 供阈值/先验使用的语义化索引
B_SUB, B_KICK, B_BASS, B_LOWMID, B_MID, B_UPPERMID, B_PRESENCE, B_TREBLE, B_HIGH, B_AIR = range(
    N_BANDS
)


def band_bin_edges(sr: int, n_fft: int) -> np.ndarray:
    """返回长度 nb+1 的 bin 边界（左闭右开），最后一个边界包含 Nyquist bin。"""
    hz_per_bin = sr / n_fft
    nb_bins = n_fft // 2 + 1
    edges = [max(1, int(round(lo / hz_per_bin))) for lo, _ in BAND_EDGES_HZ]
    edges.append(min(nb_bins, max(edges[-1] + 1, int(round(BAND_EDGES_HZ[-1][1] / hz_per_bin)) + 1)))
    edges = np.asarray(edges, dtype=np.int32)
    # 保证严格递增
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    return edges


def band_of_bin(sr: int, n_fft: int) -> np.ndarray:
    """长度 nb_bins 的查表: bin -> band 索引（-1 表示 DC，不参与任何带）。"""
    edges = band_bin_edges(sr, n_fft)
    nb_bins = n_fft // 2 + 1
    out = np.full(nb_bins, -1, dtype=np.int8)
    for b in range(N_BANDS):
        out[edges[b] : edges[b + 1]] = b
    return out
