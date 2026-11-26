"""全部可调参数集中在此，便于嵌入式端用 JSON 覆盖调参（P5 落地）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Params:
    """算法参数（每一项都标注了它的物理作用）。

    时间单位: 采样点。频率单位: Hz。
    所有 *_q88 参数为 Q8.8 对数域整数 (1.0 倍频程 = 256，即 1 个 log2 单位)。
    换算: 1.0 倍频程 = 6.02 dB  =>  Q8.8 的 1 LSB = 0.0234 dB。
    """

    # ---- 前端 ----
    sr: int = 16000          # 采样率: 决定最高可分析频率 8 kHz (= sr/2)
    win: int = 1024          # 帧长 64 ms: 决定频率分辨率 (1/64ms = 15.6 Hz/bin)
    hop: int = 256           # 跳步 16 ms: 决定时间分辨率 (62.5 帧/秒)
    # 环形缓冲 = win + hop: 保证任务读窗时 ISR 不会覆写窗内样本 (见 framer.py)

    # ---- 数值模式 ----
    fixed: bool = True       # True: 整数定点路径 (target C 等价); False: 浮点研究模式

    # ---- 子带 / flux ----
    band_max_hz: float = 8000.0   # 分析上限 (= Nyquist)
    flux_bin_diff: int = 1        # 带内取 ±1 bin 最大值: 抗频率抖动/跳 bin

    # ---- 候选检测 ----
    thr_q_num: int = 1       # 阈值分位 = thr_q_num / thr_q_den
    thr_q_den: int = 2       #   1/2 = 中位数; 调大(如 8/10)可减少误报, P2 再定
    thr_delta_q88: int = 176 # 阈值 = 分位电平 + delta。176/256 = 0.69 oct = 4.1 dB

    thr_hyst_num: int = 3    # 迟滞: 触发后必须回落到 thresh*3/5 才允许再次触发
    thr_hyst_den: int = 5    #   (防止一个"山包"上的抖动重复出事件)
    min_ioi_ms: float = 45.0 # 硬不应期: 同一个带 45 ms 内不可能真打两次
    #                          ⚠ 不能设成 120 ms: 128 BPM 的 16 分音符间隔只有 117 ms
    peak_lookahead: int = 1  # ±N 帧局部极大验证: N=1 表示"等下一帧确认是不是山顶"
    #                          这是 16 ms 的**延迟**，不是前瞻(用的都是已到达的数据)

    # ---- 直方图分位 ----
    hist_buckets: int = 16   # 16 个桶覆盖 0..2 倍频程
    hist_shift: int = 5      # 桶宽 = 2^5 Q8.8 = 0.125 oct = 1.5 dB
    hist_decay: int = 2048   # 计数总数超过此值时整体折半 (遗忘机制, 跟上音乐变化)

    # ---- 调试 ----
    trace: bool = False             # 记录每帧中间量 (仅原型/离线分析, 设备上关掉)
    events_per_band: bool = True    # True: 每带独立出事件; False: 仅最响带


    def with_(self, **kw: Any) -> "Params":
        return replace(self, **kw)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Params":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    @property
    def ring_len(self) -> int:
        """环形缓冲长度 = 窗长 + hop（防止 ISR 覆写窗内样本）。"""
        return self.win + self.hop

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    @property
    def min_ioi_frames(self) -> int:
        return max(1, int(round(self.min_ioi_ms * 1e-3 * self.sr / self.hop)))
