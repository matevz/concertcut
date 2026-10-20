from concertpub import interactive, programme as prog
from concertpub.analyze import Match
from concertpub.segment import Region


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def make_programme():
    return prog.Programme(items=[prog.ProgrammeItem(n, f"C{n}", subtitle=f"T{n}") for n in range(1, 6)])


def test_programme_skip_and_order():
    pr = interactive.confirm_programme(make_programme(), scripted("s", "3", "o 2 1 4 5", "y"),
                                       out=lambda *a: None)
    assert [i.number for i in pr.items] == [2, 1, 4, 5]


def test_programme_rejects_unknown_numbers():
    pr = interactive.confirm_programme(make_programme(), scripted("s 9", "s 2", ""), out=lambda *a: None)
    assert [i.number for i in pr.items] == [1, 3, 4, 5]


def test_matching_merge_assign_drop():
    pr = make_programme()
    matches = [Match(10.0 * k, 10.0 * k + 8, pr.items[k] if k < 5 else None, Region(0, 1, end_reason="applause"))
               for k in range(6)]
    out = interactive.confirm_matching(matches, pr, scripted("m 2 3", "a 1 2 4 5 +", "d 5", "yes"),
                                       out=lambda *a: None)
    assert [(m.start, m.end, m.item.number if m.item else None) for m in out] == [
        (0.0, 8.0, 1), (10.0, 28.0, 2), (30.0, 38.0, 4), (40.0, 48.0, 5)]


def test_quit_aborts():
    import pytest

    with pytest.raises(interactive.Aborted):
        interactive.confirm_programme(make_programme(), scripted("q"), out=lambda *a: None)


def test_enter_programme_retries_and_skips():
    pr = interactive.enter_programme("no programme PDF found",
                                     scripted("1 A - X", "1 B - Y", "", "1 A - X", "2 B - Y", ""),
                                     out=lambda *a: None)
    assert [(i.number, i.composer, i.title) for i in pr.items] == [(1, "A", "X"), (2, "B", "Y")]
    assert interactive.enter_programme("x", scripted(""), out=lambda *a: None) is None


def test_reread_options_raise():
    import pytest
    from concertpub.analyze import RereadMedia, RereadProgramme

    with pytest.raises(RereadProgramme):
        interactive.confirm_programme(make_programme(), scripted("r"), out=lambda *a: None)
    with pytest.raises(RereadMedia):
        interactive.confirm_matching([], None, scripted("r"), out=lambda *a: None)


def test_matching_shows_positions_in_source_files(monkeypatch):
    from concertpub import gain
    from concertpub.analyze import SourceMap

    monkeypatch.setattr(gain, "peak_of", lambda audio, z0, dur, rate=None: -9.0)
    from concertpub.media import MediaFile

    cams = [MediaFile("/m/S1.MP4", 1000.0, start=0.0), MediaFile("/m/S2.MP4", 1000.0, start=1000.0)]
    rec = [MediaFile("/m/ZOOM-0001.WAV", 3000.0, start=0.0)]
    smap = SourceMap(cams, rec, audio_start=-600.0)  # recorder started 10 min before the camera
    m = Match(990.0, 1250.0, None, Region(0, 1, end_reason="applause"), offset=-600.0, sources=smap)
    assert m.spans() == {
        "camera": [{"file": "S1.MP4", "from": 990.0, "to": 1000.0}, {"file": "S2.MP4", "from": 0.0, "to": 250.0}],
        "recorder": [{"file": "ZOOM-0001.WAV", "from": 1590.0, "to": 1850.0}]}
    lines = []
    interactive.show_matching([m], None, out=lines.append)
    text = "\n".join(lines)
    assert "S1.MP4 16:30.0 - S2.MP4 4:10.0" in text
    assert "ZOOM-0001.WAV  26:30.0 - 30:50.0" in text


def test_camera_sound_only_shows_camera():
    from concertpub.analyze import SourceMap
    from concertpub.media import MediaFile

    cams = [MediaFile("/m/S1.MP4", 4000.0, start=0.0)]
    m = Match(3700.0, 3720.0, None, Region(0, 1), sources=SourceMap(cams, list(cams), 0.0))
    assert list(m.spans()) == ["camera"]
    assert interactive._clock(3700.0) == "1:01:40.0"


def test_back_option_raises():
    import pytest
    from concertpub.analyze import BackToProgramme

    with pytest.raises(BackToProgramme):
        interactive.confirm_matching([], None, scripted("b"), out=lambda *a: None)


def test_lead_in_shown_and_changed():
    ms = [Match(10.0, 100.0, None, Region(0, 1, end_reason="applause"), first_note=13.0),
          Match(101.0, 200.0, None, Region(0, 1, end_reason="applause"), first_note=103.5)]
    lines = []
    interactive.show_matching(ms, None, out=lines.append)
    assert "3.0 s" in lines[2] and "2.5 s" in lines[3]
    out = interactive.confirm_matching(ms, None, scripted("l 1 1.5", "l 2 5", "y"), out=lambda *a: None)
    assert out[0].start == 11.5 and out[0].lead_in == 1.5
    assert out[1].start == 100.0  # limited by the end of the previous piece
    out = interactive.confirm_matching(out, None, scripted("l * 0", ""), out=lambda *a: None)
    assert [m.start for m in out] == [13.0, 103.5]


def test_numbering_notes():
    assert interactive.numbering_notes([1, 2, 3]) == []
    assert interactive.numbering_notes([1, 2, 4, 5]) == ["missing: 3 (not performed?)"]
    assert interactive.numbering_notes([1, 3, 2]) == ["not in order: 3 before 2"]
    assert interactive.numbering_notes([1, 2, 2, 3]) == ["used more than once: 2"]
    lines = []
    pr = make_programme()
    pr.items = [pr.items[0], pr.items[2], pr.items[1]]
    interactive.show_programme(pr, out=lines.append)
    assert any("not in order: 3 before 2" in l for l in lines)


def test_gain_shown_and_set(monkeypatch):
    from concertpub import gain
    from concertpub.analyze import SourceMap, concert_gain
    from concertpub.media import MediaFile

    peaks = {0.0: -12.0, 100.0: -8.75, 200.0: -10.0}  # by recorder start
    monkeypatch.setattr(gain, "peak_of", lambda audio, z0, dur, rate=None: peaks[round(z0, 1)])
    smap = SourceMap([], [MediaFile("/m/r.wav", 400.0)], 0.0, target_dbfs=-0.1)
    ms = [Match(100.0 * k, 100.0 * k + 90, None, Region(0, 1), offset=0.0, sources=smap) for k in range(3)]
    g, loudest, by_user = concert_gain(ms)
    assert round(g, 2) == 8.65 and loudest is ms[1] and not by_user
    lines = []
    interactive.show_matching(ms, None, out=lines.append)
    text = "\n".join(lines)
    assert "-8.8 dB*" in text and "-12.0 dB " in text and "Concert gain: +8.65 dB (automatic" in text

    out = interactive.confirm_matching(ms, None, scripted("g 10", "y"), out=lambda *a: None)
    assert concert_gain(out)[:3:2] == (10.0, True)
    lines = []
    interactive.show_matching(out, None, out=lines.append)
    assert any("WARNING: with this gain, piece(s) 2 will clip." in l for l in lines)
    out = interactive.confirm_matching(out, None, scripted("g auto", "y"), out=lambda *a: None)
    assert round(concert_gain(out)[0], 2) == 8.65



def test_kdenlive_option_raises():
    import pytest
    from concertpub.analyze import PreviewKdenlive

    with pytest.raises(PreviewKdenlive):
        interactive.confirm_matching([], None, scripted("k"), out=lambda *a: None)
