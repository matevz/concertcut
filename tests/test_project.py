import json
import xml.etree.ElementTree as ET
from fractions import Fraction

import opentimelineio as otio

from concertcut import kdenlive, otio_export
from concertcut.media import MediaFile
from concertcut.plan import Piece, Plan, safe_filename


def make_plan(tmp_path) -> Plan:
    cam = [MediaFile(f"/media/c/S00{i}.MP4", 1357.96, start=1357.96 * (i - 1), audio_index=1, video_index=0,
                     sample_rate=48000, channels=2, fps=Fraction(50), width=1920, height=1080)
           for i in (1, 2, 3)]
    rec = [MediaFile("/media/c/ZOOM_LR-0001.WAV", 3728.02, start=0.0, audio_index=0, sample_rate=96000,
                     channels=2, bits=24),
           MediaFile("/media/c/ZOOM_LR-0002.WAV", 600.0, start=3728.02, audio_index=0, sample_rate=96000,
                     channels=2, bits=24)]
    pieces = [
        Piece(1, "Uršula Jašovec", "Grajska gospa in berač", 500.0, 820.5),
        Piece(2, "Andrej Misson", "Mlinar in smrt", 900.0, 1340.0, audio_offset_ms=1.5),
        # spans both recorder files and two camera files
        Piece(3, "Vitja Avsec", "Grofovo nasilje nad mlinarjem", 3700.0, 3800.0),
        Piece(4, "X", "disabled", 3900.0, 3950.0, enabled=False),
    ]
    plan = Plan(concert={"name": "20261016_test", "dir": str(tmp_path)}, video=cam, audio=rec,
                audio_start=40.0, pieces=pieces, path=tmp_path / "p.json")
    plan.settings.gain_db = 25.0
    return plan


def test_kdenlive_structure(tmp_path):
    plan = make_plan(tmp_path)
    path = kdenlive.write(plan, tmp_path / "x.kdenlive")
    root = ET.parse(path).getroot()
    assert root.tag == "mlt" and root.get("producer") == "main_bin"
    ids = [e.get("id") for e in root if e.get("id")]
    assert len(ids) == len(set(ids)), "duplicate ids"
    # every referenced producer is defined before it is used
    seen = set()
    for el in root:
        for ref in el.iter():
            if ref.tag in ("entry", "track") and ref.get("producer"):
                assert ref.get("producer") in seen, ref.get("producer")
        seen.add(el.get("id"))
    seq = [t for t in root.findall("tractor") if t.get("id", "").startswith("{")][0]
    props = {p.get("name"): p.text for p in seq.findall("property")}
    guides = json.loads(props["kdenlive:sequenceproperties.guides"])
    assert [g["comment"] for g in guides] == [
        "01 Uršula Jašovec - Grajska gospa in berač", "01 end",
        "02 Andrej Misson - Mlinar in smrt", "02 end",
        "03 Vitja Avsec - Grofovo nasilje nad mlinarjem", "03 end"]
    assert guides[0]["pos"] == 25000 and guides[1]["pos"] == 41025
    # recorder track: both Zoom files whole, back to back at the median piece offset (40 s)
    rec_pl = root.find("playlist[@id='playlist0']")
    assert [c.tag for c in rec_pl if c.tag != "property"] == ["blank", "entry", "entry"]
    assert rec_pl.find("blank").get("length") == "00:00:40.000"
    entries = rec_pl.findall("entry")
    assert [(e.get("in"), e.get("out")) for e in entries] == [
        ("00:00:00.000", "01:02:08.000"), ("00:00:00.000", "00:09:59.980")]
    levels = [e.find("filter/property[@name='level']").text for e in entries]
    assert levels == ["00:00:00.000=25.00", "00:00:00.000=25.00"]  # one constant concert gain
    assert entries[0].find("filter/property[@name='max_gain']").text == "27dB"
    # camera sound muted, video track hides audio
    tractors = {t.get("id"): t for t in root.findall("tractor")}
    seq_tracks = [t.get("producer") for t in seq.findall("track")]
    assert seq_tracks[0] == "producer0" and len(seq_tracks) == 5
    assert {t.get("hide") for t in tractors[seq_tracks[2]].findall("track")} == {"both"}
    assert {t.get("hide") for t in tractors[seq_tracks[3]].findall("track")} == {"audio"}
    mb = root.find("playlist[@id='main_bin']")
    mbp = {p.get("name"): p.text for p in mb.findall("property")}
    assert mbp["kdenlive:docproperties.version"] == "1.1"
    assert mbp["kdenlive:docproperties.profile"] == "atsc_1080p_50"
    assert len(mb.findall("entry")) == 1 + 5


def test_otio(tmp_path):
    plan = make_plan(tmp_path)
    path = otio_export.write(plan, tmp_path / "x.otio")
    tl = otio.adapters.read_from_file(str(path))
    assert [m.name.split(" [")[0] for m in tl.tracks.markers] == [p.basename for p in plan.enabled_pieces()]
    kinds = [t.kind for t in tl.tracks]
    assert kinds.count("Audio") == 2 and kinds.count("Video") == 1


def test_plan_roundtrip(tmp_path):
    plan = make_plan(tmp_path)
    plan.save()
    again = Plan.load(plan.path)
    assert again.to_dict() == plan.to_dict()
    assert again.fps == Fraction(50)


def test_safe_filename():
    assert safe_filename('03 A: B/C - "Ivan, moj sin?"') == "03 A BC - Ivan, moj sin"
    assert Piece(9, "", "Encore", 0, 1).basename == "09 Encore"


def test_recorder_started_before_camera(tmp_path):
    """A piece beginning 30 s before the camera was switched on: project shifted, no negative times."""
    plan = make_plan(tmp_path)
    plan.audio_start = -100.0  # recorder ran 100 s before the camera
    plan.pieces[0].start, plan.pieces[0].end = -30.0, 200.0
    assert plan.project_shift == 30.0
    root = ET.parse(kdenlive.write(plan, tmp_path / "early.kdenlive")).getroot()
    seq = [t for t in root.findall("tractor") if t.get("id", "").startswith("{")][0]
    guides = json.loads({p.get("name"): p.text for p in seq.findall("property")}
                        ["kdenlive:sequenceproperties.guides"])
    assert guides[0]["pos"] == 0
    rec = root.find("playlist[@id='playlist0']")
    assert rec[1].tag == "entry" and rec[1].get("in") == "00:01:10.000"  # recorder cropped to 70 s
    cam = root.find("playlist[@id='playlist4']")
    assert cam[0].tag == "blank" and cam[0].get("length") == "00:00:30.000"
    tl = otio.adapters.read_from_file(str(otio_export.write(plan, tmp_path / "early.otio")))
    assert tl.tracks.markers[0].marked_range.start_time.value == 0


def test_volume_filter_in_out_match_clip(tmp_path):
    plan = make_plan(tmp_path)
    plan.audio_start = -600.0  # recorder ran 10 min before the camera: first clip is trimmed
    root = ET.parse(kdenlive.write(plan, tmp_path / "trim.kdenlive")).getroot()
    for e in root.find("playlist[@id='playlist0']").findall("entry"):
        f = e.find("filter")
        assert (f.get("in"), f.get("out")) == (e.get("in"), e.get("out"))
        assert f.find("property[@name='level']").text == e.get("in") + "=25.00"


def test_audio_only_project(tmp_path):
    plan = make_plan(tmp_path)
    plan.video = []
    plan.audio_start = 0.0
    root = ET.parse(kdenlive.write(plan, tmp_path / "audio.kdenlive")).getroot()
    assert root.find("profile").get("frame_rate_num") == "25"  # no camera: default profile
    assert len(root.find("playlist[@id='playlist0']").findall("entry")) == 2  # both recorder files
    assert not root.find("playlist[@id='playlist4']").findall("entry")  # no video
    assert len(root.find("playlist[@id='main_bin']").findall("entry")) == 1 + 2
    tl = otio.adapters.read_from_file(str(otio_export.write(plan, tmp_path / "audio.otio")))
    assert [t.name for t in tl.tracks] == ["Recorder"]


def test_camera_sound_as_recording_project(tmp_path):
    from dataclasses import replace

    plan = make_plan(tmp_path)
    plan.audio = [replace(f) for f in plan.video]  # no recorder: the camera's own sound
    plan.audio_start = 0.0
    assert plan.camera_is_recorder
    root = ET.parse(kdenlive.write(plan, tmp_path / "cam.kdenlive")).getroot()
    chains = {c.get("id"): {p.get("name"): p.text for p in c.findall("property")} for c in root.findall("chain")}
    rec = root.find("playlist[@id='playlist0']").findall("entry")
    assert [chains[e.get("producer")]["resource"] for e in rec] == [f.path for f in plan.video]
    assert {chains[e.get("producer")]["set.test_audio"] for e in rec} == {"0"}  # audible, with volume
    assert not root.find("playlist[@id='playlist2']").findall("entry")  # no muted duplicate
    assert len(root.find("playlist[@id='main_bin']").findall("entry")) == 1 + 3  # each file once
    tl = otio.adapters.read_from_file(str(otio_export.write(plan, tmp_path / "cam.otio")))
    assert [t.name for t in tl.tracks] == ["Camera sound", "Camera"]
