"""Stage-0 normaliser tests. Every case is a real example from docs (Data Forensics doc / reading sheet)."""
from amlc.norm import house_cores, number_relation, parse_address, parse_name


# ---------- names ----------
def test_alias_markers_take_the_right_side_as_core():
    n = parse_name("Beloveo One F/K/A Platt Cornerstone Ncr Inc")
    assert n["core"] == "platt cornerstone ncr" and n["alias"] == "beloveo one"
    assert parse_name("Caloveo DBA: Pediatric Associates LLC")["core"] == "pediatric associates"
    assert parse_name("Veokelo t/a Technologies Champions Services")["core"] == "technologies champions services"
    assert parse_name("Onyxveo formerly Espinoza & Davis Industrials LLC")["core"] == "espinoza davis industrials"


def test_ids_brackets_hyphens_duplicates_and_spacing():
    assert parse_name("Smith Medical Medical (ID: 94444)")["core"] == "smith medical"
    assert parse_name("Frontier-Fund Corp")["core"] == "frontier fund"
    assert parse_name("Frontier [Fund]")["core"] == "frontier fund"
    assert parse_name("Ram Tech  Private Private Ltd")["core"] == "ram tech"


def test_suffixes_are_split_out_not_lost():
    n = parse_name("Florence E. Johnson, Esq., PC PC")
    assert n["core"] == "florence e johnson" and set(n["suffix"].split()) == {"esq", "pc"}
    assert parse_name("Palghar Services Pvt. Ltd.")["suffix"] != parse_name("Palghar Services Limited")["suffix"]
    assert parse_name("Premier Clear Spa Co")["suffix"] == parse_name("Premier Clear Spa Company")["suffix"]


def test_diacritics_homoglyphs_fold_channel():
    assert parse_name("Régional Fúnd Cápital")["core"] == "regional fund capital"
    # fold channel maps OCR look-alikes to one class so lnc/Inc, Ca1kin/Calkin, El0ise/Eloise agree
    assert parse_name("Ca1kin Xflh lNC")["fold"] == parse_name("Calkin Xflh Inc")["fold"]
    assert parse_name("El0ise Iser")["fold"] == parse_name("Eloise Iser")["fold"]


def test_domain_and_pipe_url_forms():
    n = parse_name("ELITEGROWTH.COM")
    assert n["is_domain"] == 1 and n["flat"] == "elitegrowth"
    assert parse_name("Elite Growth Ltd")["flat"] == "elitegrowth"
    assert parse_name("Behavioral Health Medicine of Washington | www.behavioral.com")["core"] == \
        "behavioral health medicine washington"


def test_honorifics_and_connectors_dropped():
    assert parse_name("M/s Aditya Properties")["core"] == "aditya properties"
    assert parse_name("Shri Ram Marketing")["core"] == "ram marketing"
    assert parse_name("Ets Art SASU")["core"] == "art"


# ---------- house numbers ----------
def test_house_number_structured_forms():
    assert house_cores("0014048") == {"14048"}
    assert house_cores("##1723") == {"1723"}
    assert "6968" in house_cores("6968-6972")
    assert house_cores("3707-D") == {"3707"} and house_cores("8871C") == {"8871"}
    assert "2/3" in house_cores("F-02/3") and "2/3" in house_cores("F-2/3")
    assert "130" in house_cores("1-30")


def test_number_relation_structured_vs_bare():
    assert number_relation({"128"}, {"28"}) == "digit_drop"
    assert number_relation({"3979"}, {"979"}) == "digit_drop"
    assert number_relation({"14048"}, {"14048"}) == "equal"
    assert number_relation({"720"}, {"920"}) == "substitution"
    assert number_relation({"1138"}, {"1149"}) == "substitution"
    assert number_relation(set(), {"128"}) == "missing"


# ---------- addresses ----------
def test_address_slots_us():
    a = parse_address("6646  SHANNON LN, MENTOR, PO BOX 4945, OH", "US")
    assert a["house"] == "6646" and a["street"] == "shannon" and a["street_type"] == "lane"
    assert a["state"] == "oh" and a["city"] == "mentor" and a["unit"] == "4945"
    b = parse_address("6646 Shannon Lane, PMB 9222, Mentor, Ohio", "US")
    assert (b["house"], b["street"], b["state"], b["city"]) == ("6646", "shannon", "oh", "mentor")


def test_address_shuffled_components_and_place_suffixes():
    a = parse_address("WTERLOO CITY, IA, ##2715 RAGLEY DR", "US")
    assert a["house"] == "2715" and a["street"] == "ragley" and a["city"] == "wterloo" and a["state"] == "ia"


def test_generator_bad_expansion_first_saint():
    a = parse_address("215 FIRST SAINT, SPRINGFIELD, IL", "US")
    b = parse_address("215 1st Street, Springfield, Illinois", "US")
    assert a["street"] == b["street"] and a["street_type"] == b["street_type"] == "street"


def test_india_door_prefixes_and_native_state():
    a = parse_address("PLOT NO 0689, NEW DELHI, WEST DELHI, दिल्ली", "India")
    b = parse_address("H.NO 689, New Delhi, Delhi", "India")
    assert a["house"] == b["house"] == "689"
    assert a["state"] == b["state"] == "delhi"


def test_null_family_and_missing():
    a = parse_address("C/O NARAYAN, WASHIM, <NULL>, Maharashtra", "India")
    assert "null" not in a["tokens"].split() and a["state"] == "maharashtra"
    assert parse_address("(no address)", "US")["has_addr"] == 0
    assert parse_address("", "US")["has_addr"] == 0


def test_france_rue_and_numero():
    a = parse_address("N° 16 R DE LA MATTE, SAINT-NAZAIRE", "France")
    assert a["house"] == "16" and a["street_type"] == "rue" and a["street"] == "matte"
    b = parse_address("16 Rue de la Matte, Saint-Nazaire, Pays de la Loire", "France")
    assert b["street"] == "matte" and b["city"] == a["city"] == "saint nazaire"


# ---------- Indian-script words mapped to English (learned dictionary, resources/in_xlit.json)
def test_native_script_name_becomes_english_words():
    from amlc.norm import XLIT
    if not XLIT["name"]:
        return   # dictionary not built in this checkout
    n = parse_name("शक्ति अर्बन प्रोडक्ट्स प्राइवेट लिमिटेड")
    assert n["core"] == "shakti urban products" and set(n["suffix"].split()) == {"pvt", "ltd"}
    assert parse_name("जैन वेंचर्स प्राइवेट लिमिटेड")["key"] == parse_name("Jain Ventures Private Limited")["key"]


def test_latin_words_are_never_remapped():
    assert parse_name("Al Noor Traders")["core"] == "al noor traders"
