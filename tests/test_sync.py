import numpy as np

from concertpub import sync
from concertpub.media import MediaFile, segments


def test_xcorr_lag_finds_shift():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(20000)
    a = b[3000:9000] + 0.3 * rng.standard_normal(6000)  # a starts at b[3000]
    lag = sync.xcorr_lag(a, b)
    assert lag.lag == 3000 and lag.confident


def test_xcorr_lag_negative():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(8000)
    b = a[500:]  # b starts 500 samples into a -> a starts at b[-500]
    assert sync.xcorr_lag(a, b).lag == -500


def test_gcc_phat_subsample():
    sr = 16000
    rng = np.random.default_rng(2)
    n = sr * 5
    src = rng.standard_normal(n + 4000)
    x = src[2000:2000 + n]
    # y contains x delayed by 0.1234 s within a +-0.3 s search window
    d_true = 0.1234
    k = int(round(d_true * sr))
    y = np.concatenate([rng.standard_normal(k) * 0.1, x, rng.standard_normal(int(0.6 * sr)) * 0.1])
    d, sharp = sync.gcc_phat(x, y, sr, 0.3)
    assert abs(d - k / sr) < 1e-4 and sharp > 20


def test_robust_line_ignores_outliers():
    t = np.arange(0, 5000, 100.0)
    y = 12.5 + 4e-6 * t
    y[[3, 17, 30]] += [0.2, -0.05, 0.5]
    a, b, mask = sync.robust_line(t, y, tol=0.004)
    assert abs(a - 12.5) < 1e-6 and abs(b - 4e-6) < 1e-9
    assert mask.sum() == len(t) - 3


def test_camera_placement_and_gap():
    files = [MediaFile("a.mp4", 100.0), MediaFile("b.mp4", 100.0), MediaFile("c.mp4", 50.0)]
    for f, s in zip(files, (0, 100, 230)):
        f.start = s
    # recorder time at each camera file start: recorder started 20 s after camera;
    # b follows a seamlessly (0.1 s correlation jitter), c starts after a 30 s gap
    z = [-20.0, 80.1, 210.0]
    a_start = sync.apply_camera_placement(files, z)
    assert [round(f.start, 3) for f in files] == [0.0, 100.0, 229.9]
    assert abs(a_start - 20.0) < 0.1


def test_segments_with_gaps():
    f1, f2 = MediaFile("1.wav", 10.0, start=0.0), MediaFile("2.wav", 10.0, start=12.0)
    seg = segments([f1, f2], 8.0, 15.0)
    assert [(s[0].path if s[0] else None, round(s[1], 3), round(s[2], 3)) for s in seg] == [
        ("1.wav", 8.0, 2.0), (None, 0.0, 2.0), ("2.wav", 0.0, 3.0)]
    assert segments([f1], -2.0, 1.0)[0][0] is None


def test_fine_sync_moves_misplaced_block(monkeypatch):
    """A camera file after a gap, misplaced by 40 ms, is corrected by the fine sync."""
    sr = sync.FINE_SR
    rng = np.random.default_rng(3)
    true_a = 12.0  # recorder time 0 is at timeline 12 s
    world = rng.standard_normal(int(400 * sr)).astype(np.float32)  # "sound in the hall", timeline 0..400 s
    cam = [MediaFile("a.mp4", 150.0, start=0.0), MediaFile("b.mp4", 150.0, start=150.0),
           MediaFile("c.mp4", 90.0, start=300.04)]  # c truly starts at 300.0
    true_start = {"a.mp4": 0.0, "b.mp4": 150.0, "c.mp4": 300.0}
    rec = [MediaFile("r.wav", 380.0, start=0.0)]

    def fake_decode_group(files, sr_, t0, t1, channels=1):
        n = int(round((t1 - t0) * sr_))
        out = np.zeros(n, np.float32)
        for f in files:
            a, b = max(t0, f.start), min(t1, f.end)
            if b <= a:
                continue
            if f.path.endswith(".wav"):
                w0 = a + true_a  # recorder time -> timeline
            else:
                w0 = true_start[f.path] + (a - f.start)
            i, j = int(round((a - t0) * sr_)), int(round(w0 * sr_))
            m = min(int(round((b - a) * sr_)), n - i, len(world) - j)
            out[i:i + m] = world[j:j + m]
        return out

    monkeypatch.setattr(sync.media, "decode_group", fake_decode_group)
    res = sync.fine_sync(cam, rec, coarse=12.01)
    assert abs(res.audio_start - true_a) < 1e-3
    assert abs(cam[2].start - 300.0) < 1e-3
    assert abs(res.drift) < 1e-7
