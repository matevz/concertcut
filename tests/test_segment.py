import numpy as np

from concertpub import segment
from concertpub.gain import fade_curve

FPS = 100


def build(parts):
    """parts: list of (kind, seconds). Returns scores (100 fps), energy_db, hf_db."""
    sc, en, hf = [], [], []
    for kind, secs in parts:
        n = int(secs * FPS)
        s = np.full((n, 3), 0.02, np.float32)
        e = np.full(n, -60.0)
        h = np.full(n, -70.0)
        if kind == "music":
            s[:, 0] = 0.7
            e[:] = -20
            h[:] = -40
        elif kind == "applause":
            s[:, 1] = 0.8
            e[:] = -25
            h[:] = -28
        elif kind == "speech":
            s[:, 2] = 0.8
            e[:] = -30
            h[:] = -45
        sc.append(s)
        en.append(e)
        hf.append(h)
    return np.concatenate(sc), np.concatenate(en), np.concatenate(hf)


CONCERT = [("silence", 30), ("speech", 20), ("music", 120), ("applause", 8), ("music", 60),
           ("silence", 5), ("music", 70), ("applause", 6), ("speech", 30), ("music", 90),
           ("applause", 10), ("silence", 20)]


def test_pieces_split_at_applause_and_speech_only():
    scores, energy, hf = build(CONCERT)
    total = len(energy) / FPS
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, total, expected=3)
    assert len(regions) == 3, warnings
    starts = [r.start for r in regions]
    ends = [r.end for r in regions]
    # piece 1 starts right after the speech at 50 s (no quiet to keep), ends where applause starts
    assert abs(starts[0] - 50.3) < 0.15 and abs(ends[0] - 170.0) < 0.2
    # piece 2 spans the 5 s rest: 178 .. 313
    assert abs(starts[1] - 179.5) < 0.15 and abs(ends[1] - 313.0) < 0.2
    assert abs(starts[2] - 349.3) < 0.15 and abs(ends[2] - 439.0) < 0.2
    assert [r.end_reason for r in regions] == ["applause", "applause", "applause"]
    assert not warnings


def test_split_at_silence_when_programme_needs_more_pieces():
    scores, energy, hf = build(CONCERT)
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=4)
    assert len(regions) == 4
    assert abs(regions[1].end - 242.5) < 0.3 and abs(regions[2].start - 242.7) < 0.2
    assert any("split" in w for w in warnings)


def test_extra_regions_warn():
    scores, energy, hf = build(CONCERT)
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=2)
    assert len(regions) == 3 and any("Encore" in w for w in warnings)


def test_short_regions_dropped():
    scores, energy, hf = build([("silence", 10), ("music", 10), ("applause", 5), ("music", 60),
                                ("applause", 5)])
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS)
    assert len(regions) == 1 and any("short" in w for w in warnings)


def test_reverb_tail_end_without_applause():
    parts = [("silence", 10), ("music", 60)]
    scores, energy, hf = build(parts + [("silence", 20)])
    # music decays over 2 s into silence (reverb), detector says "other" right away
    i = 70 * FPS
    energy[i:i + 200] = np.linspace(-20, -60, 200)
    regions, _, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS)
    assert regions[0].end_reason == "silence"
    assert 71.0 < regions[0].end < 72.5


def test_fade_curve():
    g = fade_curve(100, 10, 20)
    assert g[0] == 0 and g[10] == 1 and g[50] == 1 and g[-1] < 0.06 and g[-20] == 1


def test_pitch_cue_separates_pieces_without_applause():
    # piece, silence, pitch cue (4 s), silence, piece - no applause in between
    parts = [("silence", 10), ("music", 60), ("silence", 5), ("cue", 4), ("silence", 12), ("music", 50),
             ("applause", 10), ("silence", 10)]
    sc, en, hf = [], [], []
    for kind, secs in parts:
        s, e, h = build([("music" if kind == "cue" else kind, secs)])
        if kind == "cue":
            s[:, 0] = 0.55
        sc.append(s), en.append(e), hf.append(h)
    scores, energy, hf = np.concatenate(sc), np.concatenate(en), np.concatenate(hf)
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=2)
    assert len(regions) == 2, warnings
    # 3 s of quiet hall kept before the first note at 91 s
    assert abs(regions[1].start - 88.0) < 0.1 and "pitch cue" in regions[1].start_reason
    assert abs(regions[0].end - 74.5) < 0.1 and regions[0].end_reason == "next piece"  # just before the cue


def test_encore_after_ovation():
    parts = [("silence", 10)]
    for k in range(3):
        parts += [("music", 60), ("applause", 8), ("silence", 5)]
    parts[-2] = ("applause", 45)  # final ovation after piece 3
    parts += [("music", 40), ("applause", 20)]
    scores, energy, hf = build(parts)
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=3)
    assert len(regions) == 4 and [r.encore for r in regions] == [False, False, False, True]


def test_loud_chord_near_end_is_not_a_cue():
    # piece ending with: quiet, loud short chord, quiet, 10 s of music, applause
    parts = [("silence", 10), ("music", 60), ("silence", 4), ("cue", 3), ("silence", 6), ("music", 10),
             ("applause", 10), ("silence", 5)]
    sc, en, hf = [], [], []
    for kind, secs in parts:
        s, e, h = build([("music" if kind == "cue" else kind, secs)])
        if kind == "cue":
            s[:, 0] = 0.55
        sc.append(s), en.append(e), hf.append(h)
    scores, energy, hf = np.concatenate(sc), np.concatenate(en), np.concatenate(hf)
    regions, warnings, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=1)
    assert len(regions) == 1 and regions[0].end > 92.5, (regions, warnings)


def test_lead_in_shorter_when_pitch_still_rings():
    # pitch cue at 75..79 s ringing out until 89.5 s, first note at 91 s -> lead-in 1.5 s
    parts = [("silence", 10), ("music", 60), ("silence", 5), ("cue", 4), ("silence", 12), ("music", 50),
             ("applause", 10), ("silence", 10)]
    sc, en, hf = [], [], []
    for kind, secs in parts:
        s, e, h = build([("music" if kind == "cue" else kind, secs)])
        if kind == "cue":
            s[:, 0] = 0.55
        sc.append(s), en.append(e), hf.append(h)
    scores, energy, hf = np.concatenate(sc), np.concatenate(en), np.concatenate(hf)
    energy[79 * FPS:int(89.5 * FPS)] = np.linspace(-25, -50, int(10.5 * FPS))  # resonance
    regions, _, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS, expected=2)
    assert 89.4 < regions[1].start < 90.0, regions[1]


def test_soft_opening_before_detector_start():
    # quiet hall, soft singing for 7 s that the detector scores as "other", then clear music
    scores, energy, hf = build([("silence", 20), ("silence", 7), ("music", 60), ("applause", 10), ("silence", 5)])
    energy[20 * FPS:27 * FPS] = -45.0  # soft opening: audible, but not "music" for the detector
    regions, _, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS)
    assert abs(regions[0].start - 17.0) < 0.1, regions[0]  # first note at 20 s, minus 3 s lead-in


def test_start_not_in_applause_tail():
    # applause, its audible tail (labelled "other"), then the encore right away
    scores, energy, hf = build([("silence", 10), ("applause", 10), ("silence", 3), ("music", 60), ("applause", 5)])
    energy[20 * FPS:22 * FPS] = -30.0  # applause dying away, no longer recognised as applause
    regions, _, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS)
    assert regions[0].start >= 21.5, regions[0]


def test_end_before_first_isolated_claps():
    scores, energy, hf = build([("silence", 10), ("music", 60), ("silence", 3), ("applause", 10), ("silence", 5)])
    # two people clap first, at 72.0 s and 72.6 s; the full applause starts at 73.0 s
    for t in (72.0, 72.6):
        i = int(t * FPS)
        hf[i:i + 4] = -40.0
        energy[i:i + 4] = -35.0
    regions, _, _ = segment.segment(scores, energy, hf, FPS, len(energy) / FPS)
    assert regions[0].end_reason == "applause"
    assert 71.8 < regions[0].end < 72.0, regions[0].end
