import numpy as np

from onset_detect.framer import Framer


def test_no_frame_before_window_filled():
    f = Framer(1024, 256)
    assert f.push(np.zeros(1000, np.int16)) == []
    assert len(f.push(np.zeros(24, np.int16))) == 1


def test_frame_content_matches_signal():
    rng = np.random.default_rng(5)
    x = rng.integers(-3000, 3000, 16000 * 3).astype(np.int16)
    f = Framer(1024, 256)
    frames = []
    for i in range(0, len(x), 137):
        frames.extend(f.push(x[i : i + 137]))
    for k in (0, 1, 5, len(frames) - 1):
        assert np.array_equal(frames[k], x[k * 256 : k * 256 + 1024])


def test_block_size_invariance_of_framer():
    rng = np.random.default_rng(6)
    x = rng.integers(-3000, 3000, 16000).astype(np.int16)
    ref = None
    for bs in (1, 37, 128, 256, 1000, 4096):
        f = Framer(1024, 256)
        frames = []
        for i in range(0, len(x), bs):
            frames.extend(f.push(x[i : i + bs]))
        if ref is None:
            ref = frames
        else:
            assert len(frames) == len(ref)
            for a, b in zip(frames, ref):
                assert np.array_equal(a, b)


def test_wraparound_stress():
    """块长与 hop 互质 + 非整齐长度，覆盖环形缓冲回绕分支。"""
    rng = np.random.default_rng(9)
    x = rng.integers(-1000, 1000, 5000).astype(np.int16)
    f = Framer(1024, 256)
    frames = []
    for i in range(0, len(x), 97):
        frames.extend(f.push(x[i : i + 97]))
    for k, fr in enumerate(frames):
        assert np.array_equal(fr, x[k * 256 : k * 256 + 1024])


def test_reset():
    f = Framer(1024, 256)
    f.push(np.ones(2048, np.int16))
    n = f.n_frames
    assert n > 0
    f.reset()
    assert f.n_frames == 0
    assert f.push(np.zeros(1000, np.int16)) == []
