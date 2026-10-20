"""The analyze flow with the media steps faked: reuse of an existing config and re-reading."""

import json

import pytest

from concertpub import analyze as A, programme as prog, segment, sync
from concertpub.media import MediaFile


@pytest.fixture
def fake(monkeypatch, tmp_path):
    calls = {"detect": 0, "read": 0}
    video = [MediaFile(str(tmp_path / "cam.mp4"), 3000.0, video_index=0, audio_index=1, width=1920,
                       height=1080)]
    audio = [MediaFile(str(tmp_path / "rec.wav"), 3000.0, audio_index=0, sample_rate=48000, channels=2,
                       bits=24)]

    def read(concert_dir, programme_path, skip, enter):
        calls["read"] += 1
        items = [prog.ProgrammeItem(n, f"C{n}", subtitle=f"T{n}") for n in (1, 2, 3)]
        pr = prog.Programme(items=[i for i in items if not skip or i.number not in skip],
                            header=["Concert"], source="p.pdf")
        return pr, "p.pdf"

    def pieces(pr):
        regions = [segment.Region(0, 0, encore=(k == 3)) for k in range(4)]
        smap = A.SourceMap(video, audio, 5.0)
        ms = [A.Match(100.0 * k + 10, 100.0 * k + 90, None, r, sources=smap) for k, r in enumerate(regions)]
        return A._assign_in_order(ms, pr), []

    def detect(vfiles, afiles, cache, model_dir, params, pr):
        calls["detect"] += 1
        return (sync.SyncResult(5.0, 0.0, 1.0, []), *pieces(pr))

    def segment_again(vfiles, afiles, sres, cache, model_dir, params, pr):
        calls["segment"] = calls.get("segment", 0) + 1
        return pieces(pr)

    monkeypatch.setattr(A.media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(A, "_segment", segment_again)
    monkeypatch.setattr(A, "_noise", lambda *a: {"hum": None})
    monkeypatch.setattr(A, "_load_media", lambda *a: (video, audio))
    monkeypatch.setattr(A, "_read_programme", read)
    monkeypatch.setattr(A, "_detect", detect)
    monkeypatch.setattr(A.sync, "piece_offsets",
                        lambda v, a, s, ranges: [{"offset": 5.002, "anchors": 3, "spread_ms": 1.0}] * len(ranges))
    monkeypatch.setattr(A.gain, "update_gain", lambda config, known=None: None)
    return calls, tmp_path


def run(tmp_path, **kw):
    return A.analyze(tmp_path, tmp_path / "concertpub.json", **kw)


def test_first_run_then_reuse_keeps_edits(fake):
    calls, tmp = fake
    config = run(tmp)
    assert [p.basename for p in config.pieces] == ["01 C1 - T1", "02 C2 - T2", "03 C3 - T3", "04 Encore"]
    # the user edits the config by hand
    d = json.loads((tmp / "concertpub.json").read_text())
    d["pieces"][3]["title"] = "Zdravljica"
    d["pieces"][1]["enabled"] = False
    d["pieces"][0]["end"] = "00:01:35.000"
    (tmp / "concertpub.json").write_text(json.dumps(d))

    seen = {}
    config = run(tmp, confirm_programme=lambda pr: seen.setdefault("pr", pr),
               confirm_matching=lambda ms, pr: seen.setdefault("ms", ms))
    assert calls == {"detect": 1, "read": 1}  # nothing read again
    assert [i.number for i in seen["pr"].items] == [1, 2, 3]
    assert [m.title for m in seen["ms"]] == ["T1", "T2", "T3", "Zdravljica"]
    assert config.pieces[0].end == 95.0 and not config.pieces[1].enabled
    assert config.pieces[3].basename == "04 Zdravljica"
    assert config.pieces[0].audio_offset_ms == pytest.approx(2.0)  # kept, not measured again


def test_changed_programme_renames_in_order(fake):
    calls, tmp = fake
    run(tmp)

    def skip_two(pr):
        pr.items = [i for i in pr.items if i.number != 2]
        return pr

    config = run(tmp, confirm_programme=skip_two)
    assert [p.basename for p in config.pieces] == ["01 C1 - T1", "02 C3 - T3", "03 Encore", "04 Encore"]


def test_reread_programme_and_media(fake):
    calls, tmp = fake
    run(tmp)
    answers = iter([A.RereadProgramme(), None])

    def confirm_programme(pr):
        a = next(answers)
        if a:
            raise a
        return pr

    media_answers = iter([A.RereadMedia(), None])

    def confirm_matching(ms, pr):
        a = next(media_answers)
        if a:
            raise a
        return ms

    run(tmp, confirm_programme=confirm_programme, confirm_matching=confirm_matching)
    assert calls == {"detect": 2, "read": 2}


def test_deleted_config_starts_from_scratch(fake):
    calls, tmp = fake
    run(tmp)
    (tmp / "concertpub.json").unlink()
    run(tmp)
    assert calls == {"detect": 2, "read": 2}


def test_load_media_without_recorder_or_camera(monkeypatch, tmp_path):
    cams = [MediaFile(str(tmp_path / "c1.mp4"), 100.0, video_index=0, audio_index=1, sample_rate=48000)]
    recs = [MediaFile(str(tmp_path / "r1.wav"), 100.0, audio_index=0, sample_rate=96000)]
    monkeypatch.setattr(A.media, "load_group", lambda paths, kind: cams if kind == "video" else recs)

    monkeypatch.setattr(A.media, "discover", lambda d: ([tmp_path / "c1.mp4"], []))
    v, a = A._load_media(tmp_path, None, None)
    assert v == cams and [f.path for f in a] == [cams[0].path] and a[0] is not cams[0]

    monkeypatch.setattr(A.media, "discover", lambda d: ([], [tmp_path / "r1.wav"]))
    v, a = A._load_media(tmp_path, None, None)
    assert v == [] and a == recs

    monkeypatch.setattr(A.media, "discover", lambda d: ([], []))
    with pytest.raises(A.media.MediaError):
        A._load_media(tmp_path, None, None)


class Crash(Exception):
    pass


def crash(*a):
    raise Crash()


def test_programme_saved_before_slow_steps(fake):
    calls, tmp = fake

    def skip_two(pr):
        pr.items = [i for i in pr.items if i.number != 2]
        return pr

    with pytest.raises(Crash):  # crash at the matching question of the first run
        run(tmp, confirm_programme=skip_two, confirm_matching=crash)
    d = json.loads((tmp / "concertpub.json").read_text())
    assert [i["number"] for i in d["programme"]["items"]] == [1, 3] and d["pieces"] == []
    # the next run starts from the saved programme and runs the detection again
    config = run(tmp)
    assert calls == {"detect": 2, "read": 1}
    assert [p.basename for p in config.pieces][:2] == ["01 C1 - T1", "02 C3 - T3"]


def test_changed_programme_renames_saved_pieces_immediately(fake):
    calls, tmp = fake
    run(tmp)

    def skip_two(pr):
        pr.items = [i for i in pr.items if i.number != 2]
        return pr

    with pytest.raises(Crash):
        run(tmp, confirm_programme=skip_two, confirm_matching=crash)
    d = json.loads((tmp / "concertpub.json").read_text())
    assert [(p["composer"], p["title"]) for p in d["pieces"]] == [
        ("C1", "T1"), ("C3", "T3"), ("", "Encore"), ("", "Encore")]


def scripted(*steps):
    """Callback answering with the given steps: an exception to raise, or a function."""
    it = iter(steps)

    def cb(*args):
        step = next(it)
        if isinstance(step, Exception):
            raise step
        return step(*args)
    return cb


def test_back_with_unchanged_programme_keeps_matching(fake):
    calls, tmp = fake

    def merge_first_two_then_go_back(ms, pr):
        ms[0:2] = [A.replace(ms[0], end=ms[1].end, offset=None)]
        raise A.BackToProgramme()

    config = run(tmp, confirm_programme=scripted(lambda pr: pr, lambda pr: pr),
               confirm_matching=scripted(merge_first_two_then_go_back, lambda ms, pr: ms))
    assert len(config.pieces) == 3 and config.pieces[0].end == 190.0  # the merge survived going back
    assert calls["detect"] == 1 and "segment" not in calls


def test_back_and_change_order_renames_without_new_analysis(fake):
    calls, tmp = fake

    def reverse(pr):
        pr.items = list(reversed(pr.items))
        return pr

    config = run(tmp, confirm_programme=scripted(lambda pr: pr, reverse),
               confirm_matching=scripted(A.BackToProgramme(), lambda ms, pr: ms))
    assert [p.basename for p in config.pieces][:3] == ["01 C3 - T3", "02 C2 - T2", "03 C1 - T1"]
    assert calls["detect"] == 1 and "segment" not in calls


def test_back_and_skip_finds_pieces_again(fake):
    calls, tmp = fake

    def skip_two(pr):
        pr.items = [i for i in pr.items if i.number != 2]
        return pr

    config = run(tmp, confirm_programme=scripted(lambda pr: pr, skip_two),
               confirm_matching=scripted(A.BackToProgramme(), lambda ms, pr: ms))
    assert calls["detect"] == 1 and calls["segment"] == 1
    assert [p.basename for p in config.pieces][:2] == ["01 C1 - T1", "02 C3 - T3"]
    assert [i["number"] for i in json.loads((tmp / "concertpub.json").read_text())["programme"]["items"]] == [1, 3]


def test_gain_set_in_matching_question_is_saved_and_shown_again(fake):
    calls, tmp = fake

    def set_gain(ms, pr):
        ms[0].sources.gain_db = 7.0
        return ms

    config = run(tmp, confirm_matching=set_gain)
    assert (config.settings.gain_db, config.settings.gain_db_auto) == (7.0, False)
    seen = {}
    run(tmp, confirm_matching=lambda ms, pr: seen.setdefault("gain", ms[0].sources.gain_db) and ms)
    assert seen["gain"] == 7.0

    def back_to_auto(ms, pr):
        ms[0].sources.gain_db = None
        return ms

    config = run(tmp, confirm_matching=back_to_auto)
    assert config.settings.gain_db_auto


def test_kdenlive_preview_from_matching_question(fake):
    calls, tmp = fake
    seen = {}

    def preview_then_confirm(ms, pr):
        if "previewed" not in seen:
            seen["previewed"] = True
            raise A.PreviewKdenlive()
        seen["config_saved_before_confirm"] = bool(json.loads((tmp / "concertpub.json").read_text())["pieces"])
        return ms

    run(tmp, confirm_matching=preview_then_confirm)
    project = tmp / f"{tmp.name}.concertpub.kdenlive"
    assert project.exists() and project.with_suffix(".otio").exists()
    assert "01 C1 - T1" in project.read_text()  # the guides of the current matching
    assert seen["config_saved_before_confirm"] is False  # concertpub.json waits for the confirmation
    # a second preview keeps the first project as .bak
    seen.clear()
    run(tmp, confirm_matching=preview_then_confirm)
    assert project.with_name(project.name + ".bak").exists()


def test_hand_edited_concert_tags_are_kept(fake):
    calls, tmp = fake
    run(tmp)
    d = json.loads((tmp / "concertpub.json").read_text())
    d["concert"].update(album="Bolečina in lepota", date="2026-10-16", artist="Komorni zbor De profundis Kranj",
                        conductor="Branka Potočnik Krajnik")
    (tmp / "concertpub.json").write_text(json.dumps(d, ensure_ascii=False))
    plan = run(tmp)
    assert plan.concert["album"] == "Bolečina in lepota" and plan.concert["date"] == "2026-10-16"
    assert plan.concert["conductor"] == "Branka Potočnik Krajnik"  # extra fields survive too
    assert run(tmp, artist="Someone else").concert["artist"] == "Someone else"  # --artist still wins
