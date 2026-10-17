from concertcut import interactive, programme as prog
from concertcut.analyze import Match
from concertcut.segment import Region


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
    from concertcut.analyze import RereadMedia, RereadProgramme

    with pytest.raises(RereadProgramme):
        interactive.confirm_programme(make_programme(), scripted("r"), out=lambda *a: None)
    with pytest.raises(RereadMedia):
        interactive.confirm_matching([], None, scripted("r"), out=lambda *a: None)
