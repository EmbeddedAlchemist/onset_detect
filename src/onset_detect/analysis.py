"""离线驱动：把整段音频按固定块长送进 OnsetDetector（与在线路径完全同一条代码）。

务必保持"离线也走流式入口"，这样离线调参的结果才等于设备上的行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import Event, OnsetDetector
from .io import iter_blocks, load_wav_mono
from .params import Params


def analyze_array(
    x: np.ndarray,
    p: Params = Params(),
    block: int = 256,
    trace: bool = True,
) -> tuple[list[Event], dict[str, np.ndarray] | None]:
    """对整段 int16 音频做检测，返回 (事件, trace)。"""
    det = OnsetDetector(p.with_(trace=trace))
    events: list[Event] = []
    for blk in iter_blocks(x, block):
        events.extend(det.process_block(blk))
    events.extend(det.flush())
    events.sort(key=lambda e: (e.sample, e.band))

    tr = None
    if trace and det.trace is not None:
        tr = {
            "band_L": np.asarray(det.trace["band_L"], dtype=np.int32),
            "band_flux": np.asarray(det.trace["band_flux"], dtype=np.int32),
            "band_thr": np.asarray(det.trace["band_thr"], dtype=np.int32),
            "frame_center": np.arange(det.n_frames) * p.hop + p.win // 2,
        }
    return events, tr


def analyze_file(
    path: str | Path,
    p: Params = Params(),
    block: int = 256,
    trace: bool = True,
) -> tuple[list[Event], dict[str, np.ndarray] | None, np.ndarray, int]:
    x, sr = load_wav_mono(path)
    if sr != p.sr:
        raise ValueError(f"采样率不匹配: 文件 {sr} Hz, 参数 {p.sr} Hz")
    events, tr = analyze_array(x, p, block=block, trace=trace)
    return events, tr, x, sr
