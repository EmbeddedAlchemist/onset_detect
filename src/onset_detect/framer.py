"""1280 点环形缓冲 + 跳帧调度。

两个基本概念（非信号处理背景必读）
================================
    "帧(frame)"：一次分析所用的连续采样点，本项目 1024 点 = 64 ms。
    "跳步(hop)"：相邻两帧之间的间隔，本项目 256 点 = 16 ms。
                 => 相邻帧重叠 768 点（75%），所以每秒产生 62.5 帧。

为什么 ring_len = win + hop
---------------------------
任务在"块结束"时刻读取最近 win 个样本 [w-win, w)；同时 ISR 正在写入 [w, w+hop)。
若 ring_len == win，则 [w, w+hop) 恰好覆写最旧的 hop 个样本（它们仍在当前窗内），
造成数据撕裂。多留一个 hop 的间隙即可做到：单生产者/单消费者、零拷贝、零锁、
且不引入任何额外延迟。

（对比常见做法: 快照 memcpy 需要额外 2 KB 缓冲 + 一次 1024 点拷贝的抢占时间。）
"""


from __future__ import annotations

import numpy as np


class Framer:
    """把任意块长的 int16 流切成固定 hop 的帧。

    >>> f = Framer(1024, 256)
    >>> frames = f.push(np.zeros(1000, np.int16))   # 未填满窗 -> 不出帧
    >>> len(frames)
    0
    """

    __slots__ = ("win", "hop", "_ring", "_w", "_since", "_total", "n_frames")

    def __init__(self, win: int, hop: int):
        if hop <= 0 or win <= 0:
            raise ValueError("win/hop 必须为正")
        self.win = int(win)
        self.hop = int(hop)
        self._ring = np.zeros(self.win + self.hop, dtype=np.int16)
        self._w = 0
        self._since = 0
        self._total = 0
        self.n_frames = 0

    def reset(self) -> None:
        self._ring[:] = 0
        self._w = 0
        self._since = 0
        self._total = 0
        self.n_frames = 0

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        """写入一块样本，返回本次产生的帧（每个都是 win 点的拷贝）。"""
        x = np.asarray(x, dtype=np.int16).ravel()
        L = self._ring.shape[0]
        out: list[np.ndarray] = []
        i = 0
        nx = x.shape[0]
        while i < nx:
            n = min(nx - i, self.hop - self._since)
            end = self._w + n
            if end <= L:
                self._ring[self._w : end] = x[i : i + n]
            else:
                first = L - self._w
                self._ring[self._w :] = x[i : i + first]
                self._ring[: end - L] = x[i + first : i + n]
            self._w = end % L
            self._since += n
            self._total += n
            i += n
            if self._since == self.hop:
                self._since = 0
                if self._total >= self.win:
                    out.append(self._window())
                    self.n_frames += 1
        return out

    def _window(self) -> np.ndarray:
        L = self._ring.shape[0]
        start = (self._w - self.win) % L
        end = start + self.win
        if end <= L:
            return self._ring[start:end].copy()
        first = L - start
        buf = np.empty(self.win, dtype=np.int16)
        buf[:first] = self._ring[start:]
        buf[first:] = self._ring[: end - L]
        return buf

    @property
    def frame_center(self) -> int:
        """最近一帧中心的绝对采样位置（用于事件时间戳）。"""
        return (self.n_frames - 1) * self.hop + self.win // 2
