"""流式、定点、面向嵌入式目标的鼓点检测。

模块分工（与 target C 实现一一对应）:

    fxp.py        整数定点原语 (Q15 / Q8.8 / Q4.12), 无 float / 无 sqrt / 无除法
    fft_fixed.py  1024 点 block-floating-point Q15 实数 FFT + Hann 窗
    framer.py     1280 点环形缓冲 + 跳帧调度 (零拷贝, 单生产者/单消费者)
    bands.py      10 个对数子带的 bin 边界表 (Flash 常量)
    quantile.py   16 桶整数直方图分位数跟踪 (自适应阈值参考电平)
    detector.py   ★唯一流式入口: process_block(int16[]) -> [Event]
    io.py         WAV 读取 / 单声道下混 / 分块读取

约束（详见 docs/architecture.md）:
    - 输入 16000 Hz / 16 bit / 单声道
    - 每 16 ms 一帧 1024 点, 环大小 1024+256 = 1280 点 (2.5 KB)
    - 运行时路径不使用 float、不使用 sqrt、不使用除法、不做动态分配
    - 不前瞻: 所有运算只使用已到达的样本; 判别用"延迟决策 + 特征环"
"""

from .detector import Event, OnsetDetector
from .envelope import estimate_tempo_bpm, instrument_envelopes
from .hits import Hit, HitMerger, group_hits, hit_label
from .params import Params

__all__ = [
    "Event",
    "OnsetDetector",
    "Params",
    "Hit",
    "HitMerger",
    "group_hits",
    "hit_label",
    "instrument_envelopes",
    "estimate_tempo_bpm",
]
__version__ = "0.1.0"
