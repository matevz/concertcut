from dataclasses import replace

from concertcut import render
from test_project import make_plan


def test_no_video_skips_mp4(tmp_path, capsys):
    plan = make_plan(tmp_path)
    plan.video = []
    render.render(plan, tmp_path / "out", dry_run=True)
    out = capsys.readouterr().out
    assert out.count(".flac") == 3 and ".mp4" not in out


def test_camera_sound_flac_uses_camera_audio_stream(tmp_path):
    plan = make_plan(tmp_path)
    plan.audio = [replace(f) for f in plan.video]
    plan.audio_start = 0.0
    cmd = render.flac_command(plan, plan.pieces[0], tmp_path / "x.flac")
    assert plan.video[0].path in cmd and "[0:1]" in " ".join(cmd)
    assert "s32" in cmd  # lossy camera sound: written as 24-bit FLAC
    mp4 = " ".join(render.mp4_command(plan, plan.pieces[0], tmp_path / "x.mp4"))
    assert "[0:0]" in mp4 and "[1:1]" in mp4  # picture and sound both from the camera file
