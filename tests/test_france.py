"""France-only normalisation (real test rows). US/India behaviour must not change."""
from amlc.clean import clean_address
from amlc.norm import number_relation, parse_address, parse_name


def fr(a):
    return parse_address(a, "France")


def test_bis_ter_are_part_of_the_house_number():
    a = fr("5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine")
    b = fr("5B Rue Pierre Dignac, La Teste-de-Buch, Gironde")
    c = fr("5 Rue André Dignac, La Teste-de-Buch, Nouvelle-Aquitaine")
    assert a["house"] == b["house"] == "5b" and c["house"] == "5"
    assert a["street"] == b["street"] == "pierre dignac"
    assert number_relation({"5b"}, {"5b"}) == "equal"
    assert number_relation({"5b"}, {"5"}) == "substitution"      # not digit_drop
    assert fr("1 TER Rue Kléber, Lille")["house"] == fr("1 T R. KLEBER, Lille")["house"] == "1t"
    assert fr("61bis RUE du Coudray, Nantes")["house"] == fr("61 B RUE DU COUDRAY, Nantes")["house"] == "61b"


def test_range_and_rue_abbreviation_are_not_letters():
    assert fr("66 R Van Dyck, Pas-de-Calais, Calais")["house"] == "66"
    assert fr("10 à 12 Rue Nationale, Lille")["house"] == "10"


def test_departments_map_to_regions():
    assert fr("5B Rue Pierre Dignac, La Teste-de-Buch, Gironde")["state"] == "nouvelle aquitaine"
    assert fr("66 R Van Dyck, Pas-de-Calais, Calais")["state"] == "hauts de france"
    assert fr("12 Ch. Des Garennes, Lille, Nord")["state"] == "hauts de france"
    assert fr("25B RUE DE LA FONTAINE AUX BRETONS, Pornic, Loire-Atlantique")["state"] == "pays de la loire"


def test_new_street_words_and_saint():
    r = fr("9 B CITÉ MOUNEYRA, BORDEAUX, Nouvelle-Aquitaine")
    assert (r["house"], r["street"], r["street_type"], r["city"]) == ("9b", "mouneyra", "cite", "bordeaux")
    r = fr("5 Alle De Tourny, Bordeaux")
    assert (r["street"], r["street_type"], r["city"]) == ("tourny", "allee", "bordeaux")
    assert fr("12 Ch. Des Garennes, Lille, Nord")["street_type"] == "chemin"
    assert fr("24 Aveue De L'armistice, Pessac")["street_type"] == "avenue"
    r = fr("NO. 12 RUE DE L'ABBESSE, St Nazaire")
    assert r["city"] == "saint nazaire" and r["street_type"] == "rue"
    assert fr("12 Rue du Dr Schweitzer, Lille")["street"] == "dr schweitzer"


def test_nums_keep_the_letter_in_france_only():
    assert clean_address("5 bis Rue Pierre Dignac", "France")[1] == clean_address("5B RUE PIERRE DIGNAC", "France")[1] == "5b"
    assert clean_address("5 Rue Pierre Dignac", "France")[1] == "5"
    assert clean_address("3707-D Main St", "US")[1] == clean_address("3707-D Main St")[1]


def test_names():
    assert parse_name("Ets Martin")["key"] == parse_name("Etablissements Martin")["key"] == "martin"
    assert parse_name("Martin & Cie")["key"] == parse_name("Martin Compagnie")["key"] == "martin"
    assert parse_name("Dupont EI", "France")["key"] == "dupont" and parse_name("Dupont EI", "France")["suffix"] == "ei"
    assert parse_name("Dupont EI")["key"] == "dupont ei"                      # US/India unchanged


def test_us_india_unchanged():
    assert parse_address("5425 152nd Avenue, Ramsey, MN", "US")["house"] == "5425"
    assert parse_address("159a Main St, Tyler, TX", "US")["house"] == "159"
    assert parse_address("12 St Marks Pl, New York, NY", "US")["street_type"] == "place"
    assert number_relation({"159"}, {"1590"}) == "digit_drop"
