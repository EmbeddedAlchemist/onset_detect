import numpy as np
import pytest

from onset_detect.fxp import LOG2_LUT, log2_q8_8, log_mag_q8_8, mul_q15, q88_to_db, sat16


def test_lut_shape():
    assert LOG2_LUT.shape == (17,)
    assert LOG2_LUT[0] == 0 and LOG2_LUT[16] == 256
    assert np.all(np.diff(LOG2_LUT) > 0)


def test_log2_accuracy_over_full_range():
    x = np.unique(np.round(np.logspace(0, 9.2, 5000)).astype(np.int64))
    x = np.concatenate([x, np.array([1, 2, 3, 255, 256, 257, 511, 512, 1023, 2**30])])
    got = log2_q8_8(x) / 256.0
    ref = np.log2(x)
    # 实测最大误差 0.0044 oct = 0.038 dB（LUT 取整 0.002 + 插值 0.0007 + 12 bit 尾数截断）
    assert np.max(np.abs(got - ref)) < 5e-3




def test_log2_semantics_lsb():
    # 1 LSB 应为 1/256 倍频程
    assert log2_q8_8(2) == 256
    assert log2_q8_8(4) == 512
    assert abs(int(log2_q8_8(3)) - 405) <= 1  # log2(3)*256 = 405.75


def test_log_mag_no_sqrt():
    rng = np.random.default_rng(1)
    re = rng.integers(-20000, 20000, 5000).astype(np.int32)
    im = rng.integers(-20000, 20000, 5000).astype(np.int32)
    keep = (re != 0) | (im != 0)
    re, im = re[keep], im[keep]
    got = log_mag_q8_8(re, im) / 256.0
    ref = np.log2(np.hypot(re.astype(np.float64), im.astype(np.float64)))
    assert np.max(np.abs(got - ref)) < 4e-3



def test_mul_q15_truncation():
    # 与 C 的 (a*b) >> 15 完全一致（向负无穷取整）
    assert mul_q15(np.int32(16384), np.int32(16384)) == 8192
    assert mul_q15(np.int32(-1), np.int32(1)) == -1
    assert mul_q15(np.int32(32767), np.int32(32767)) == 32766


def test_sat16():
    x = np.array([-40000, -32768, 0, 32767, 40000], dtype=np.int32)
    assert sat16(x).tolist() == [-32768, -32768, 0, 32767, 32767]


def test_q88_db_roundtrip():
    for db in (-24.0, -6.0, 0.0, 3.0, 12.0):
        q = round(db * 256 / (20 * np.log10(2)))
        assert abs(float(q88_to_db(q)) - db) < 0.03
