from concertpub import cli


def test_folder_alone_runs_the_whole_flow(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "cmd_analyze", lambda a: calls.append(("analyze", a.concert_dir, a.yes)) or "config")
    monkeypatch.setattr(cli, "cmd_project", lambda a, config=None: calls.append(("kdenlive", config)))
    monkeypatch.setattr(cli, "cmd_render", lambda a, config=None: calls.append(("render", config)))
    assert cli.main([str(tmp_path), "--yes"]) == 0
    assert calls == [("analyze", tmp_path, True), ("kdenlive", "config"), ("render", "config")]
    calls.clear()
    assert cli.main(["-v", str(tmp_path)]) == 0  # options before the folder
    assert calls[0][:2] == ("analyze", tmp_path)


def test_subcommands_unchanged(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "cmd_render", lambda a, config=None: calls.append("render"))
    assert cli.main(["render", str(tmp_path)]) == 0 and calls == ["render"]

