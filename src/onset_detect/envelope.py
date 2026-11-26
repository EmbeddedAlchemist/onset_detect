"""把 10 条带 flux 合成几条"乐器强度包络"（onset strength envelope）。

为什么是包络，而不是事件点
========================
下游的节拍跟踪/预测（DP、Kalman、PLL）最想要的输入是**连续强度曲线**：

    离散事件点                        连续包络
    ---------------------------      ----------------------------
    只有"发生/没发生"                 带"有多强"，可以加权
    漏检 = 信息完全丢失               漏检时邻域能量还能撑住相位
    同一击多带触发会重复计数          取 max 天然只算一次
    多乐器只能靠标签区分              每条曲线单独喂一个跟踪器

所以本项目对外的"标准输出"不是事件列表，而是**每乐器的强度包络 + 事件**
（事件用于打点/调试，包络用于节拍）。

三条通道（P0 粗版；P3 用 HPSS + 判别特征升级）
    pulse : 低频打击 = max(sub, kick)，可减去一部分 bass 以压制贝斯音符
    snare : 中高频宽带 = max(presence, treble)
    hat   : 高频 = max(high, air)

注意 pulse_net 的"减去 bass"只是**log 域的一阶近似**（见 §4：dB 做差=比值），
它会同时压掉"底鼓+贝斯同时出现"的真鼓（掩蔽场景），
这正是 P3 要用 HPSS 替换它的原因。这里保留它就是为了量化这个取舍。
"""

from __future__ import annotations

import numpy as np

from .bands import B_AIR, B_BASS, B_HIGH, B_KICK, B_PRESENCE, B_SUB, B_TREBLE


def instrument_envelopes(
    band_flux_q88: np.ndarray,
    bass_suppress_num: int = 1,
    bass_suppress_den: int = 2,
) -> dict[str, np.ndarray]:
    """输入 (帧数, 10) 的带 flux（Q8.8），输出几条同长度的强度包络。

    Parameters
    ----------
    band_flux_q88 : (n_frames, 10) int32
    bass_suppress_num/den : pulse_net 里扣掉多少 bass 带 flux（1/2 = 扣一半）

    Returns
    -------
    dict: pulse / pulse_net / snare / hat，均为 int32 Q8.8
    """
    f = np.asarray(band_flux_q88, dtype=np.int32)
    pulse = np.maximum(f[:, B_SUB], f[:, B_KICK])
    bass = (f[:, B_BASS] * int(bass_suppress_num)) // int(bass_suppress_den)
    pulse_net = np.maximum(0, pulse - bass)
    snare = np.maximum(f[:, B_PRESENCE], f[:, B_TREBLE])
    hat = np.maximum(f[:, B_HIGH], f[:, B_AIR])
    return {"pulse": pulse, "pulse_net": pulse_net, "snare": snare, "hat": hat}


def estimate_tempo_bpm(
    env: np.ndarray,
    frame_rate: float,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
) -> tuple[float, float]:
    """在强度包络上做自相关，估一个参考 BPM。返回 (bpm, 周期性强度 0..1)。

    这只是**离线自检**，用来验证"这条包络是否已经足以承载节拍信息"。
    真正的在线节拍跟踪会做 DP / Kalman，不在本函数里。
    """
    e = np.asarray(env, dtype=np.float64)
    e = e - e.mean()
    n = e.shape[0]
    if n < int(2 * frame_rate):
        return 0.0, 0.0
    # FFT 自相关（比直接 O(n^2) 快得多）
    spec = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(spec.real**2 + spec.imag**2)[:n]
    ac /= ac[0] + 1e-12

    lag_lo = max(1, int(60.0 * frame_rate / bpm_max))
    lag_hi = min(n - 2, int(60.0 * frame_rate / bpm_min))
    if lag_hi <= lag_lo:
        return 0.0, 0.0
    seg = ac[lag_lo:lag_hi]
    lag = lag_lo + int(np.argmax(seg))

    # 抛物线插值到亚帧精度
    if 0 < lag < n - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag_f = lag + delta
    bpm = 60.0 * frame_rate / lag_f

    # 周期性强度：峰值相对局部中位数的显著性
    local = np.median(seg)
    strength = float((bpm > 0) * max(0.0, (ac[lag] - local) / (1.0 - local + 1e-12)))
    return float(bpm), strength
