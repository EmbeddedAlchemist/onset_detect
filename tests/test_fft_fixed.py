"""定点 FFT 与 numpy 浮点 FFT 的对拍。

定点路径的标定关系:
    X_int ~= 16384 * X_float * 2^(-shift)
    L = 256*log2|X_int| - shift*256 + RFFT_LOG_BIAS_Q88   (= 256*log2|X_float|)
其中 X_float = rfft(x_int16 * hann_normalized)。
"""

import numpy as np
import pytest

from onset_detect.fft_fixed import (
    RfftTables,
    hann_q15,
    log_spectrum_q88,
    rfft_log_offset,
    rfft_q15,
)

from onset_detect.fxp import log2_sqrt_q8_8


def _fixed_log(x_int16: np.ndarray, t: RfftTables, n: int):
    w = hann_q15(n).astype(np.int32)
    xw = (x_int16.astype(np.int32) * w) >> 15
    s, sh = rfft_q15(xw, t)
    L = log2_sqrt_q8_8(s) + rfft_log_offset(int(sh))
    return L, int(sh), w



def _float_log(x_int16: np.ndarray, w: np.ndarray, n: int):
    ref = np.fft.rfft(x_int16.astype(np.float64) * (w.astype(np.float64) / 32767.0))
    return 256.0 * np.log2(np.abs(ref) + 1e-12)


def test_hann_q15_shape():
    w = hann_q15(1024)
    assert w.dtype == np.int16
    assert w[0] == 0 and w[1024 // 2] == 32767
    assert np.all(w >= 0)


def test_rfft_matches_numpy_on_noise():
    n = 1024
    t = RfftTables(n)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(n) * 6000).astype(np.int16)
    L, sh, w = _fixed_log(x, t, n)
    ref = _float_log(x, w, n)

    err_oct = (L - ref) / 256.0
    # 只在显著 bin 上比较（低电平 bin 受整数截断影响）
    mask = ref > ref.max() - 60 * 256 / (20 * np.log10(2))
    assert mask.sum() > 400
    assert np.max(np.abs(err_oct[mask])) < 0.05
    assert abs(float(np.mean(err_oct[mask]))) < 0.02


def test_rfft_bias_stable_across_levels():
    """标定常数必须与信号电平无关（block-floating-point 的 shift 补偿是否正确）。

    注: 极低电平(如 -48 dBFS 以下)时, 各级 >>1 的截断会产生最多 ~0.2 dB 的
    电平相关偏置; 该偏置远小于 3-6 dB 的检测阈值, 且是缓慢变化的, 不会产生假 onset。
    因此这里只在实用电平范围内要求稳定。
    """
    n = 1024
    t = RfftTables(n)
    rng = np.random.default_rng(11)
    base = rng.standard_normal(n)
    biases = []
    for gain in (1024.0, 4096.0, 12000.0, 24000.0):
        x = np.clip(base * gain, -32767, 32767).astype(np.int16)
        L, _, w = _fixed_log(x, t, n)
        ref = _float_log(x, w, n)
        mask = ref > ref.max() - 30 * 256 / (20 * np.log10(2))
        ref_db = ref[mask] / 256.0
        biases.append(float(np.mean((L[mask] / 256.0) - ref_db)))
    assert max(biases) - min(biases) < 0.01, biases




def test_rfft_single_tone_bin():
    """满幅单音应落在正确的 bin，且电平正确。"""
    n = 1024
    sr = 16000
    t = RfftTables(n)
    f = 60.0
    k = f * n / sr
    idx = np.arange(n)
    x = np.round(20000.0 * np.sin(2 * np.pi * f * idx / sr)).astype(np.int16)
    L, _, w = _fixed_log(x, t, n)
    peak = int(np.argmax(L[1:]) + 1)
    assert abs(peak - round(k)) <= 1
    ref = _float_log(x, w, n)
    assert abs(float(L[peak] - ref[peak]) / 256.0) < 0.05


def test_rfft_batch_equals_single():
    n = 1024
    t = RfftTables(n)
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((5, n)) * 3000).astype(np.int16)
    w = hann_q15(n).astype(np.int32)
    xw = (x.astype(np.int32) * w) >> 15
    s_batch, sh_batch = rfft_q15(xw, t)
    for i in range(5):
        s_i, sh_i = rfft_q15(xw[i], t)
        assert int(sh_i) == int(sh_batch[i])
        assert np.array_equal(s_i, s_batch[i])


def test_log_spectrum_shape():
    n = 1024
    t = RfftTables(n)
    w = hann_q15(n).astype(np.int32)
    x = np.zeros(n, dtype=np.int32)
    L, sh = log_spectrum_q88(x, t)
    assert L.shape == (n // 2 + 1,)
