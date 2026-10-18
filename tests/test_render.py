from dataclasses import replace

from concertcut import render
from test_project import make_config


def test_no_video_skips_mp4(tmp_path, capsys):
    config = make_config(tmp_path)
    config.video = []
    render.render(config, tmp_path / "out", dry_run=True)
    out = capsys.readouterr().out
    assert out.count(".flac") == 3 and ".mp4" not in out


def test_camera_sound_flac_uses_camera_audio_stream(tmp_path):
    config = make_config(tmp_path)
    config.audio = [replace(f) for f in config.video]
    config.audio_start = 0.0
    cmd = render.flac_command(config, config.pieces[0], tmp_path / "x.flac")
    assert config.video[0].path in cmd and "[0:1]" in " ".join(cmd)
    assert cmd[cmd.index("-sample_fmt") + 1] == "s16"
    assert "aresample=44100" in " ".join(cmd)  # 48 kHz camera sound -> CD quality
    mp4 = " ".join(render.mp4_command(config, config.pieces[0], tmp_path / "x.mp4"))
    assert "[0:0]" in mp4 and "[1:1]" in mp4  # picture and sound both from the camera file


def test_flac_is_cd_quality_with_maximum_compression(tmp_path):
    config = make_config(tmp_path)  # 96 kHz / 24-bit recorder
    cmd = render.flac_command(config, config.pieces[0], tmp_path / "x.flac")
    filt = cmd[cmd.index("-filter_complex") + 1]
    assert cmd[cmd.index("-compression_level") + 1] == "12" and cmd[cmd.index("-sample_fmt") + 1] == "s16"
    assert "aresample=44100:resampler=soxr" in filt and "dither_method=triangular" in filt
    assert filt.index("volume=") < filt.index("aresample=44100")  # gain first, then CD conversion


def test_flac_rate_never_above_the_recording(tmp_path):
    from concertcut.config import flac_rate
    from concertcut.media import MediaFile

    assert flac_rate([MediaFile("a.wav", 1, sample_rate=96000)]) == 44100
    assert flac_rate([MediaFile("a.wav", 1, sample_rate=32000)]) == 32000
    assert flac_rate([MediaFile("a.wav", 1, sample_rate=96000)], override=48000) == 48000


def test_mp4_audio_cd_rate_video_unchanged(tmp_path):
    config = make_config(tmp_path)
    cmd = render.mp4_command(config, config.pieces[0], tmp_path / "x.mp4")
    filt = cmd[cmd.index("-filter_complex") + 1]
    assert "aresample=44100:resampler=soxr" in filt and "aresample=48000" not in filt
    assert cmd[cmd.index("-c:a") + 1] == "aac" and cmd[cmd.index("-b:a") + 1] == "320k"
    assert cmd[cmd.index("-crf") + 1] == "22" and "fps=50/1" in filt  # Kdenlive quality, camera resolution and frame rate


def test_all_flac_before_any_mp4(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    order = []

    def fake_run(cmd, out, threads=None):
        order.append(out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return 0.0

    monkeypatch.setattr(render, "_run", fake_run)
    monkeypatch.setattr(render.gain, "update_gain", lambda config, known=None: None)
    config.path = None
    render.render(config, tmp_path / "out", jobs=4)
    assert order == [".flac"] * 3 + [".mp4"] * 3
