from amlc.clean import clean_address, clean_name


def test_name_drops_legal_words_junk_and_punctuation():
    assert clean_name("Moreno & Kline Industrials L.L.C.") == "moreno kline industrials"
    assert clean_name("*** Silver Lagóon") == "silver lagoon"
    assert clean_name("... QZ Logistics Pvt. Limited") == "qz logistics"


def test_name_keeps_bracketed_words():
    assert clean_name("Grace [Chape1]") == "grace chape1"
    assert clean_name("A Holdings (Global)") == "a holdings global"


def test_name_transliterates_indian_scripts_and_drops_transliterated_legal_words():
    assert clean_name("ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್") == "lots marketimg"


def test_name_handles_french_legal_words():
    assert clean_name("SASU Atelier Àmis (France)") == "atelier amis france"


def test_address_expands_abbreviations_and_extracts_numbers():
    words, nums = clean_address("#317 MAPLE CREST WAY, TROY, OH")
    assert words == "maple crest way troy oh" and nums == "317"
    words, nums = clean_address("4218 147TH AVENUE, RAMSEY, MN")
    assert nums == "4218 147"
    assert clean_address("2 HAWTHORNE CIR, POUGHKEEPSIE, NY")[0] == "hawthorne circle poughkeepsie ny"


def test_address_strips_leading_zeros_nulls_and_hashes():
    words, nums = clean_address("EVERETT, 004700 112TH PL, WA")
    assert nums == "4700 112" and words == "everett place wa"
    words, nums = clean_address("C/O SHARMA, WASHIM, <NULL>, Maharashtra")
    assert "null" not in words.split()


def test_empty_address():
    assert clean_address("") == ("", "")


def test_house_numbers_with_letters_keep_their_digits():
    assert clean_address("238B ORCHARD DRIVE, THOMASVILLE, NC")[1] == "238"
    assert clean_address("POCKET A14, BLOCK A")[1] == "14"


def test_indian_r_slash_o_is_not_rue():
    assert "rue" not in clean_address("R/O B-512 3RD FL, NEW DELHI")[0].split()
