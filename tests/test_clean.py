from amlc.clean import clean_address, clean_name


def test_name_drops_legal_words_junk_and_punctuation():
    assert clean_name("Espinoza & Davis Industrials L.L.C.") == "espinoza davis industrials"
    assert clean_name("*** Blue Hypnósis") == "blue hypnosis"
    assert clean_name("... XV Transports Pvt. Limited") == "xv transports"


def test_name_keeps_bracketed_words():
    assert clean_name("Christ [Chape1]") == "christ chape1"
    assert clean_name("A Environmental (Strategic)") == "a environmental strategic"


def test_name_transliterates_indian_scripts_and_drops_transliterated_legal_words():
    assert clean_name("ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್") == "lots marketimg"


def test_name_handles_french_legal_words():
    assert clean_name("SASU Soutien Àmis (France)") == "soutien amis france"


def test_address_expands_abbreviations_and_extracts_numbers():
    words, nums = clean_address("#185 LOCUST VIEW WAY, TROY, OH")
    assert words == "locust view way troy oh" and nums == "185"
    words, nums = clean_address("5425 152ND AVENUE, RAMSEY, MN")
    assert nums == "5425 152"
    assert clean_address("2 MULDOWNEY CIR, POUGHKEEPSIE, NY")[0] == "muldowney circle poughkeepsie ny"


def test_address_strips_leading_zeros_nulls_and_hashes():
    words, nums = clean_address("EVERETT, 001300 100TH PL, WA")
    assert nums == "1300 100" and words == "everett place wa"
    words, nums = clean_address("C/O NARAYAN, WASHIM, <NULL>, Maharashtra")
    assert "null" not in words.split()


def test_empty_address():
    assert clean_address("") == ("", "")


def test_house_numbers_with_letters_keep_their_digits():
    assert clean_address("114B RIDGEWAY DRIVE, THOMASVILLE, NC")[1] == "114"
    assert clean_address("POCKET A14, BLOCK A")[1] == "14"


def test_indian_r_slash_o_is_not_rue():
    assert "rue" not in clean_address("R/O A-308 3RD FL, NEW DELHI")[0].split()
