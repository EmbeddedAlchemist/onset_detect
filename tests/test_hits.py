"""打击合并与乐器包络的行为测试。

重点: 流式 HitMerger 与离线 group_hits 必须给出完全相同的结果
（否则"离线调参 == 设备行为"不成立）。
"""

import numpy as np

from onset_detect.analysis import analyze_array
from onset_detect.detector import Event
from onset_detect.envelope import estimate_tempo_bpm, instrument_envelopes
from onset_detect.hits import HitMerger, group_hits
from onset_detect.params import Params
from tests.test_block_invariance import _click_track


def _mk(t, band, s=100):
    return Event(t=t, sample=int(t * 16000), frame=int(t * 62.5), band=band, strength_q88=s)


def test_group_merges_bands_of_same_hit():
    ev = [_mk(1.000, 0), _mk(1.016, 1), _mk(1.032, 7), _mk(1.200, 0)]
    hits = group_hits(ev, window_s=0.040)
    assert len(hits) == 2
    assert hits[0].bands == (0, 1, 7)
    assert hits[0].n_bands == 3
    assert hits[0].pattern == "##" + "." * 5 + "#" + "." * 2
    assert hits[1].bands == (0,)


def test_streaming_merger_matches_batch():
    ev = [_mk(0.50 + 0.031 * i, i % 10) for i in range(40)]
    batch = group_hits(ev, window_s=0.040)
    m = HitMerger(window_s=0.040)
    out = []
    for e in ev:
        out.extend(m.push([e], now_s=e.t))
    out.extend(m.flush())
    assert [(h.t, h.bands) for h in out] == [(h.t, h.bands) for h in batch]


def test_hit_keeps_max_strength_per_band():
    ev = [_mk(1.000, 0, s=50), _mk(1.010, 0, s=200), _mk(1.020, 1, s=80)]
    h = group_hits(ev, window_s=0.040)[0]
    assert h.strength_q88 == (200, 80)


def test_envelopes_pick_expected_bands():
    f = np.zeros((3, 10), dtype=np.int32)
    f[0, 0] = 100  # sub
    f[1, 1] = 150  # kick
    f[2, 9] = 90   # air
    e = instrument_envelopes(f)
    assert e["pulse"].tolist() == [100, 150, 0]
    assert e["hat"].tolist() == [0, 0, 90]
    assert e["snare"].tolist() == [0, 0, 0]


def test_tempo_estimate_on_click_track():
    x = _click_track(dur=6.0)
    ev, tr = analyze_array(x, Params(trace=True), block=256)
    assert tr is not None
    env = instrument_envelopes(tr["band_flux"])["pulse"]
    bpm, strength = estimate_tempo_bpm(env, Params().frame_rate)
    # 合成鼓点间隔 0.45 s -> 133 BPM（允许自相关取到 2 倍/0.5 倍）
    assert 60 < bpm < 300
    assert strength >= 0.0
