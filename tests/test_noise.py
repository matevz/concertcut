
import numpy as np
from scipy.signal import welch

from concertpub import noise

SR = noise.SR


def hall(seconds=20.0, mains=50.0, harmonics=(1, 3, 5, 7, 9, 17, 19), level=0.003, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.002 * rng.standard_normal(len(t))  # the hall's background
    for k in harmonics:
        x += level * np.sin(2 * np.pi * mains * k * t + k)
    return x.astype(np.float32)


def spectrum(x):
    f, p = welch(x, fs=SR, nperseg=SR * 2)
    return f, 10 * np.log10(p + 1e-20)


def test_finds_50_hz_buzz_with_its_harmonics():
    hum = noise.find_hum(*spectrum(hall()))
    assert hum and hum["mains_hz"] == 50.0
    assert [round(h["hz"]) for h in hum["harmonics"]] == [50, 150, 250, 350, 450, 850, 950]
    assert abs(hum["fundamental_hz"] - 50.0) < 0.01
    assert "50 Hz mains hum (50.00 Hz) with 7 harmonic(s) up to 950 Hz" in noise.describe(hum)


def test_finds_60_hz_buzz():
    assert noise.find_hum(*spectrum(hall(mains=60.0)))["mains_hz"] == 60.0


def test_no_buzz_in_plain_background():
    assert noise.find_hum(*spectrum(hall(level=0.0))) is None



def test_quiet_windows_are_the_quietest():
    e = np.full(100 * 600, -30.0)  # 10 minutes of music at 100 fps
    e[100 * 200:100 * 210] = -70.0  # a 10 s pause
    starts = noise.quiet_windows(e, n=3)
    assert all(200 <= t <= 208 for t in starts)



def test_drifted_mains_frequency_is_fitted():
    hum = noise.find_hum(*spectrum(hall(mains=50.05)))
    assert abs(hum["fundamental_hz"] - 50.05) < 0.01
    assert abs(hum["harmonics"][-1]["hz"] - 19 * 50.05) < 0.2  # notch centred on the real 950.95 Hz


def test_mains_frequency_fitted_per_stretch():
    concert = noise.find_hum(*spectrum(hall(mains=50.01)))
    assert abs(noise.fit_fundamental(*spectrum(hall(mains=50.05, seed=1)), concert) - 50.05) < 0.01


def _buzz(seconds, f0_of_t, ks, sr):
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0_of_t(t)) / sr
    rng = np.random.default_rng(0)
    music = 0.05 * np.sin(2 * np.pi * 1234.0 * t)
    noise_floor = 0.002 * rng.standard_normal(len(t))
    return (noise_floor + sum(0.003 * np.sin(k * phase) for k in ks) + music).astype(np.float32)


def _level(x, sr, centre, width=8):
    f, p = welch(x, fs=sr, nperseg=sr)
    return 10 * np.log10(p[(f > centre - width) & (f < centre + width)].sum())


def test_dehum_one_filter_for_the_whole_piece_keeps_the_music():
    sr, ks = 16000, [1, 3, 5, 19, 41, 65]
    x = _buzz(12.0, lambda t: np.full_like(t, 50.02), ks, sr)
    hum = {"mains_hz": 50.0, "fundamental_hz": 50.02, "harmonics": [{"hz": 50.02 * k, "db": 20.0} for k in ks],
           "strongest_db": 20.0}
    y = noise.dehum(x, sr, hum)
    for k in ks:
        assert _level(x, sr, k * 50.02) - _level(y, sr, k * 50.02) > 15, k
    assert abs(_level(y, sr, 1234, 3) - _level(x, sr, 1234, 3)) < 0.2  # the music is untouched


def test_notches_cover_the_drift_measured_in_the_silences():
    sr, ks = 16000, [1, 19, 41]
    # the mains frequency wanders between 50.00 and 50.04 Hz; the silences showed +-0.02 Hz
    x = _buzz(12.0, lambda t: 50.02 + 0.02 * np.sin(2 * np.pi * t / 6.0), ks, sr)
    hum = {"mains_hz": 50.0, "fundamental_hz": 50.02, "drift_hz": 0.02,
           "harmonics": [{"hz": 50.02 * k, "db": 20.0} for k in ks], "strongest_db": 20.0}
    y = noise.dehum(x, sr, hum)
    for k in ks:
        assert _level(x, sr, k * 50.02) - _level(y, sr, k * 50.02) > 10, k


def test_chunks_give_the_same_result_as_one_pass():
    sr, ks = 16000, [1, 3, 5]
    x = _buzz(70.0, lambda t: np.full_like(t, 50.0), ks, sr)
    hum = {"mains_hz": 50.0, "fundamental_hz": 50.0, "harmonics": [{"hz": 50.0 * k, "db": 20.0} for k in ks],
           "strongest_db": 20.0}
    chunked = noise.dehum(x, sr, hum)  # 30 s chunks
    whole = noise.dehum(x, sr, hum, chunk_s=1000.0)
    assert np.max(np.abs(chunked - whole)) < 1e-4


def test_dehum_parallel_gives_the_same_result():
    x = hall(seconds=14.0)
    hum = noise.find_hum(*spectrum(x))
    assert np.array_equal(noise.dehum(x, SR, hum, workers=1), noise.dehum(x, SR, hum, workers=4))


def test_no_buzz_added_before_a_sudden_note():
    """A note starting right next to a notch frequency must not make the filter ring before it."""
    sr = 16000
    t = np.arange(int(4.0 * sr)) / sr
    hum = {"mains_hz": 50.0, "fundamental_hz": 50.0, "drift_hz": 0.03,
           "harmonics": [{"hz": 50.0 * k, "db": 20.0} for k in range(1, 30)], "strongest_db": 20.0}
    rng = np.random.default_rng(0)
    x = 0.001 * rng.standard_normal(len(t)) + sum(0.002 * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 30))
    x = (x + np.where(t >= 2.0, 0.2 * np.sin(2 * np.pi * 498.0 * t), 0.0)).astype(np.float32)
    y = noise.dehum(x, sr, hum)
    before = slice(int(1.7 * sr), int(2.0 * sr))
    rms = lambda s: 20 * np.log10(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
    assert rms(y[before]) < rms(x[before]) - 10  # quieter before the note, not a new buzz
