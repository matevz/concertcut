from concertcut import programme as prog

# Layout like the 2024 programme: no blank line between some items, a folk song
# without composer first, footnote asterisks, bio at the end.
TEXT_2024 = """\
Pravljice in bajke
v slovenski ljudski pesmi
Komorni zbor De profundis Kranj
Branka Potočnik Krajnik, dirigentka

PROGRAM
BOG DAJ, DOBER VEČER
(stara kresna pesem iz Adlešičev)
Patrick Quaggiato (1983)
RIBA FARONIKA
(ljudska iz Podmelca, Tolminsko)

Emil Adamič (1877-1936)
VRAGOVA NEVESTA
(ljudsko besedilo)

Andrej Misson (1960)
MLINAR IN SMRT*
(ljudska iz Iga pri Ljubljani, Dolenjska)

ODMOR

Uršula Jašovec (1986)
TRI ŽENE IZTRGAJO MLADENIČU SRCE*
(ljudska iz Prekmurja in Štajerske)

*krstna izvedba
Komorni zbor De profundis deluje v Kranju od jeseni 1990. Njegov
zvok je v preteklosti sooblikovalo preko sto pevk in pevcev, danes pa v
skupnem muziciranju uživa komorna zasedba pevk in pevcev, ki ostajajo
zavezani glasbeni ustvarjalnosti in poustvarjalnosti na visoki umetniški
"""

# Layout like the 2026 programme: composer with colon, CAPS incipit, ballad name, note.
TEXT_2026 = """\
BOLEČINA IN LEPOTA
Pripovedi o človeku, kakršen je bil in kakršen ostaja.
Komorni zbor De profundis Kranj
Pripovedovalka: Mija Stopar

PROGRAM
Uršula Jašovec (1986):
BOŽEC PO RIM PETLE
Grajska gospa in berač
Priredba ljudske pesmi iz Globeli pri Sodražiči na Dolenjskem

Andrej Misson (1960):
ŠPANCIRA V MALNU MALNAR MLAD
Mlinar in smrt
Priredba ljudske pesmi iz Iga pri Ljubljani na Dolenjskem

Andrej Misson (1960):
NA LOV MI PRIDE KRALJIČ MLAD
Pridobitev ljubega z navidezno smrtjo (Mlada Zora)
Ljudsko besedilo iz idrijskih hribov

Uršula Jašovec (1986)
IVAN, MOJ SIN, PA KOD SI HODO TI?
Tri žene iztrgajo mladeniču srce
Priredba ljudske pesmi iz Prekmurja in Štajerske
Komorni zbor De profundis deluje v Kranju od jeseni 1990. Njegov zvok je v
preteklosti sooblikovalo preko sto pevk in pevcev, danes pa v skupnem muziciranju
uživa komorna zasedba pevk in pevcev, ki ostajajo zavezani glasbeni ustvarjalnosti in
"""


def names(pr):
    return [f"{i.number:02d} {i.composer_or_folk} - {i.title}" for i in pr.items]


def test_2024_layout():
    pr = prog.parse_text(TEXT_2024)
    assert names(pr) == [
        "01 Traditional - Bog daj, dober večer",
        "02 Patrick Quaggiato - Riba faronika",
        "03 Emil Adamič - Vragova nevesta",
        "04 Andrej Misson - Mlinar in smrt",
        "05 Uršula Jašovec - Tri žene iztrgajo mladeniču srce",
    ]
    assert pr.items[3].premiere and pr.items[4].premiere and not pr.items[0].premiere
    assert pr.items[2].years == "1877-1936"
    assert pr.items[0].notes == ["(stara kresna pesem iz Adlešičev)"]
    assert pr.intermission_after == [4]
    assert pr.legend == ["*krstna izvedba"]
    assert pr.concert_title == "Pravljice in bajke"
    assert pr.performer == "Komorni zbor De profundis Kranj"


def test_2026_layout_uses_ballad_name():
    pr = prog.parse_text(TEXT_2026)
    assert names(pr) == [
        "01 Uršula Jašovec - Grajska gospa in berač",
        "02 Andrej Misson - Mlinar in smrt",
        "03 Andrej Misson - Pridobitev ljubega z navidezno smrtjo (Mlada Zora)",
        "04 Uršula Jašovec - Tri žene iztrgajo mladeniču srce",
    ]
    assert pr.items[1].title_caps == "ŠPANCIRA V MALNU MALNAR MLAD"
    assert pr.items[1].notes == ["Priredba ljudske pesmi iz Iga pri Ljubljani na Dolenjskem"]
    assert pr.concert_title == "Bolečina in lepota"


def test_one_line_composer_and_title():
    pr = prog.parse_text("PROGRAM\nJacobus Gallus (1550-1591): ECCE QUOMODO MORITUR\n"
                         "Anton Foerster (1837–1926): VASOVALEC\n")
    assert names(pr) == ["01 Jacobus Gallus - Ecce quomodo moritur", "02 Anton Foerster - Vasovalec"]
    assert pr.items[1].years == "1837–1926"


def test_sentence_case():
    assert prog.sentence_case("ŠPANCIRA V MALNU MALNAR MLAD") == "Špancira v malnu malnar mlad"
    assert prog.sentence_case("IVAN, MOJ SIN, PA KOD SI HODO TI?") == "Ivan, moj sin, pa kod si hodo ti?"
    assert prog.sentence_case("Mlinar in smrt") == "Mlinar in smrt"


def test_no_items_raises():
    import pytest

    with pytest.raises(prog.ProgrammeError):
        prog.parse_text("just some text\nwithout a programme\n")


def test_parse_entered_lines():
    pr = prog.parse_entered(["01 Andrej Misson - Mlinar in smrt", "",
                             "2. Vitja Avsec - Možitev deklice - hlevarja z gospodom",
                             "Bog daj, dober večer", "07 Matej Kastelic - Smrt sv. Katarine"])
    assert names(pr) == ["01 Andrej Misson - Mlinar in smrt",
                         "02 Vitja Avsec - Možitev deklice - hlevarja z gospodom",
                         "03 Traditional - Bog daj, dober večer",
                         "07 Matej Kastelic - Smrt sv. Katarine"]


def test_parse_entered_rejects_duplicates_and_empty():
    import pytest

    with pytest.raises(prog.ProgrammeError):
        prog.parse_entered(["1 A - X", "1 B - Y"])
    with pytest.raises(prog.ProgrammeError):
        prog.parse_entered(["", "  "])
