"""块大小不变性：在线/离线必须给出逐位相同的事件序列。

这是"离线调参 == 设备行为"的前提，也是移植到 C 之前必须锁死的性质。
"""

import numpy as np

from onset_detect.analysis import analyze_array
from onset_detect.detector import OnsetDetector
from onset_detect.params import Params


def _click_track(sr=16000, dur=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    x = rng.standard_normal(n) * 30.0
    for t in (0.30, 0.75, 1.20, 1.65, 2.10, 2.55):
        s = int(t * sr)
        ln = int(0.12 * sr)
        if s >= n:
            continue
        ln = min(ln, n - s)
        tt = np.arange(ln) / sr
        env = np.exp(-tt * 60.0)
        # 60 Hz 下滑到 45 Hz 的 thump + 宽带 click
        f = 60.0 * np.exp(-tt * 20.0) + 45.0
        thump = 9000.0 * env * np.sin(2 * np.pi * np.cumsum(f) / sr)
        click = np.zeros(ln)
        nc = min(40, ln)
        click[:nc] = rng.standard_normal(nc) * 6000.0
        x[s : s + ln] += thump + click
    return np.clip(x, -32767, 32767).astype(np.int16)



def test_detector_block_size_invariance():
    x = _click_track()
    ref = None
    for bs in (37, 128, 256, 1000, len(x)):
        p = Params(trace=False)
        det = OnsetDetector(p)
        ev = []
        for i in range(0, len(x), bs):
            ev.extend(det.process_block(x[i : i + bs]))
        ev.extend(det.flush())
        key = [(e.sample, e.band, e.strength_q88) for e in ev]
        if ref is None:
            ref = key
        else:
            assert key == ref, f"block={bs} 与参考不一致"


def test_detector_finds_known_clicks():
    x = _click_track()
    ev, tr = analyze_array(x, Params(trace=True), block=256)
    assert len(ev) > 5
    # 低频带 (sub/kick/bass) 上应命中 6 个合成鼓点
    low = [e for e in ev if e.band <= 2]
    hits = set()
    for e in low:
        for k, t in enumerate((0.30, 0.75, 1.20, 1.65, 2.10, 2.55)):
            if abs(e.t - t) < 0.06:
                hits.add(k)
    assert len(hits) >= 5


def test_trace_shapes():
    x = _click_track(dur=1.0)
    ev, tr = analyze_array(x, Params(trace=True), block=256)
    assert tr is not None
    nf = tr["band_flux"].shape[0]
    assert tr["band_flux"].shape == (nf, 10)
    assert tr["band_L"].shape == (nf, 10)
    assert tr["band_thr"].shape == (nf, 10)
    assert nf == int((len(x) - Params().win) // Params().hop) + 1


def test_float_and_fixed_agree_on_events():
    """定点路径与浮点路径检出结果应高度一致（允许少量边界差异）。"""
    x = _click_track(dur=3.0)
    ev_f, _ = analyze_array(x, Params(fixed=True, trace=False), block=256, trace=False)
    ev_x, _ = analyze_array(x, Params(fixed=False, trace=False), block=256, trace=False)
    lo_f = [(round(e.t, 2), e.band) for e in ev_f if e.band <= 3]
    lo_x = [(round(e.t, 2), e.band) for e in ev_x if e.band <= 3]
    inter = len(set(lo_f) & set(lo_x))
    assert inter >= max(1, int(0.9 * min(len(lo_f), len(lo_x))))
