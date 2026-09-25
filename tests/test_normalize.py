from amlc.normalize import house_number, name_key


def test_name_key_ignores_order_case_punctuation_and_legal_words():
    k = name_key("Espinoza & Davis Industrials LLC")
    assert k == name_key("Espinoza & LLC Industrials Davis")
    assert k == name_key("Espinoza & Davis Industrials L.L.C.")
    assert k == "davis espinoza industrials"


def test_name_key_strips_accents_and_junk_prefixes():
    assert name_key("*** Blue Hypnósis") == name_key("Blue Hypnosis")


def test_name_key_indian_and_french_legal_words():
    assert name_key("XV Transports Pvt. Ltd.") == name_key("XV TRANSPORTS PRIVATE LIMITED") == "transports xv"
    assert name_key("Ets Art SASU") == name_key("Ets Art")


def test_house_number_is_first_number_digits_only():
    assert house_number("5425 152ND AVENUE, RAMSEY, MN") == "5425"
    assert house_number("MN, RAMSEY, 5425 152ND AVENUE") == "5425"
    assert house_number("B-#20, Regency Plaza") == house_number("B-20, REGENCY PLAZA") == "20"
    assert house_number("RUE DES MARTYRS, TOURCOING") == ""
