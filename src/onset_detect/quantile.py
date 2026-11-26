"""整数直方图分位数跟踪：自适应阈值的"参考电平"。

这一步在做什么
--------------
给每个带跟踪一个"最近的典型 flux 水平"，阈值 = 这个水平 + 固定增量。
这样安静段落阈值自动降低、高潮段落自动升高 —— 不需要手调一个魔法常数。

什么是分位数
-----------
把最近的 N 个 flux 值从小到大排队:
    q=0.5 -> 中位数（比它大的占一半）
    q=0.8 -> 80% 分位（比它大的占 20%）
本项目默认 q=0.5，P2 会调到 0.8/0.9（见 docs/dsp_explained.md 第 7 节）。

为什么不用滑动窗中位数
--------------------
85 帧 x 10 带的滑窗中位数需要插入排序（~85 次比较+搬移）或双堆，M3/RV32 上都太贵。
16 桶直方图 + 折叠遗忘只需 "查桶 + 加 1"，量化查询是 16 次累加比较：
    ~600 cycles/帧/10 带  => 62.5 fps 下约 0.04 M cycles/s（可忽略）

语义
----
    桶宽   = 2^shift  (Q8.8)   shift=5 -> 0.125 oct = 1.5 dB
    桶值   = 桶索引 * 2^shift + 2^(shift-1)   (桶中点)
    遗忘   = 计数总数超过 decay 时整体右移 1 位（半衰期 ~decay/nb 帧）
"""


from __future__ import annotations

import numpy as np


class HistQuantile:
    """每个带一条 16 桶直方图。"""

    __slots__ = ("nb", "buckets", "shift", "decay", "q_num", "q_den", "hist", "total", "_rows")

    def __init__(
        self,
        n_bands: int,
        buckets: int = 16,
        shift: int = 5,
        decay: int = 2048,
        q_num: int = 1,
        q_den: int = 2,
    ):
        self.nb = int(n_bands)
        self.buckets = int(buckets)
        self.shift = int(shift)
        self.decay = int(decay)
        self.q_num = int(q_num)
        self.q_den = int(q_den)
        self.hist = np.zeros((self.nb, self.buckets), dtype=np.uint16)
        self.total = 0
        self._rows = np.arange(self.nb)

    def reset(self) -> None:
        self.hist[:] = 0
        self.total = 0

    def update(self, values_q88: np.ndarray) -> None:
        """把本帧各带的 flux 计入直方图。"""
        idx = np.clip(np.asarray(values_q88, dtype=np.int32) >> self.shift, 0, self.buckets - 1)
        self.hist[self._rows, idx] += 1
        self.total += self.nb
        if self.total >= self.decay:
            self.hist >>= 1
            self.total = int(self.hist.sum())

    def level(self) -> np.ndarray:
        """返回各带的分位电平（Q8.8，桶中点）。"""
        c = np.cumsum(self.hist, axis=1, dtype=np.int32)
        tot = c[:, -1:]
        # target = ceil(q_num/q_den * tot)，用整数运算避免除法以外的浮点
        target = (self.q_num * tot + self.q_den - 1) // self.q_den
        target = np.maximum(target, 1)
        idx = (c < target).sum(axis=1).astype(np.int32)
        return idx * (1 << self.shift) + (1 << (self.shift - 1))
