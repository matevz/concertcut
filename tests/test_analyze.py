"""The analyze flow with the media steps faked: reuse of an existing plan and re-reading."""

import json

import pytest

from concertcut import analyze as A, programme as prog, segment, sync
from concertcut.media import MediaFile


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

    def detect(vfiles, afiles, cache, model_dir, params, pr):
        calls["detect"] += 1
        regions = [segment.Region(0, 0, encore=(k == 3)) for k in range(4)]
        ms = [A.Match(100.0 * k + 10, 100.0 * k + 90, None, r) for k, r in enumerate(regions)]
        return sync.SyncResult(5.0, 0.0, 1.0, []), A._assign_in_order(ms, pr), []

    monkeypatch.setattr(A.media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(A, "_load_media", lambda *a: (video, audio))
    monkeypatch.setattr(A, "_read_programme", read)
    monkeypatch.setattr(A, "_detect", detect)
    monkeypatch.setattr(A.sync, "piece_offsets",
                        lambda v, a, s, ranges: [{"offset": 5.002, "anchors": 3, "spread_ms": 1.0}] * len(ranges))
    monkeypatch.setattr(A.gain, "update_gain", lambda plan: None)
    return calls, tmp_path


def run(tmp_path, **kw):
    return A.analyze(tmp_path, tmp_path / "concertcut.json", **kw)


def test_first_run_then_reuse_keeps_edits(fake):
    calls, tmp = fake
    plan = run(tmp)
    assert [p.basename for p in plan.pieces] == ["01 C1 - T1", "02 C2 - T2", "03 C3 - T3", "04 Encore"]
    # the user edits the plan by hand
    d = json.loads((tmp / "concertcut.json").read_text())
    d["pieces"][3]["title"] = "Zdravljica"
    d["pieces"][1]["enabled"] = False
    d["pieces"][0]["end"] = "00:01:35.000"
    (tmp / "concertcut.json").write_text(json.dumps(d))

    seen = {}
    plan = run(tmp, confirm_programme=lambda pr: seen.setdefault("pr", pr),
               confirm_matching=lambda ms, pr: seen.setdefault("ms", ms))
    assert calls == {"detect": 1, "read": 1}  # nothing read again
    assert [i.number for i in seen["pr"].items] == [1, 2, 3]
    assert [m.title for m in seen["ms"]] == ["T1", "T2", "T3", "Zdravljica"]
    assert plan.pieces[0].end == 95.0 and not plan.pieces[1].enabled
    assert plan.pieces[3].basename == "04 Zdravljica"
    assert plan.pieces[0].audio_offset_ms == pytest.approx(2.0)  # kept, not measured again


def test_changed_programme_renames_in_order(fake):
    calls, tmp = fake
    run(tmp)

    def skip_two(pr):
        pr.items = [i for i in pr.items if i.number != 2]
        return pr

    plan = run(tmp, confirm_programme=skip_two)
    assert [p.basename for p in plan.pieces] == ["01 C1 - T1", "02 C3 - T3", "03 Encore", "04 Encore"]


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


def test_deleted_plan_starts_from_scratch(fake):
    calls, tmp = fake
    run(tmp)
    (tmp / "concertcut.json").unlink()
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
