"""WAV 读取 / 单声道下混 / 分块读取（只依赖 stdlib wave + numpy）。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """读取 16 bit PCM WAV，返回 (int16 单声道, 采样率)。

    立体声用 `(L+R) >> 1` 下混（移位代替除法，与 C 侧一致）。
    """
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"仅支持 16 bit PCM，当前 sampwidth={sw} B")
    x = np.frombuffer(raw, dtype="<i2")
    if ch == 1:
        return x.astype(np.int16, copy=False), sr
    x = x.reshape(-1, ch).astype(np.int32)
    mono = (x[:, 0] + x[:, 1]) >> 1
    return mono.astype(np.int16), sr


def iter_blocks(x: np.ndarray, block: int):
    """按固定块长切分（最后一块可能较短）。"""
    for i in range(0, len(x), block):
        yield x[i : i + block]


def wav_info(path: str | Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        return {
            "sr": sr,
            "channels": w.getnchannels(),
            "bits": w.getsampwidth() * 8,
            "frames": n,
            "duration_s": round(n / sr, 3),
        }
