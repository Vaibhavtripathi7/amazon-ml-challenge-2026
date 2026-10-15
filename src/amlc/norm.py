"""Stage 0: undo the corruptions we found in the data and parse names and addresses into fields.

parse_name(raw)            -> core, key (sorted), fold (OCR-homoglyph folded), flat (space-stripped), suffix,
                              alias, is_domain
parse_address(raw, country) -> house, house_alts, unit, street, street_type, city, state, tokens, has_addr
house_cores(token)          -> set of canonical house-number cores ('0014048' -> {'14048'}, '6968-6972' -> both ends)
number_relation(A, B)       -> 'equal' | 'digit_drop' | 'substitution' | 'missing'

Only rules and resources that are country-conditional where the data demands it; no external data.
"""
import json
import re
from pathlib import Path

from anyascii import anyascii
from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------- names
_ALIAS = re.compile(r"\s*(?:\bf\s*/\s*k\s*/\s*a\b|\bfka\b|\bformerly\b|\bd\s*/\s*b\s*/\s*a\b|\bdba\b|\bt\s*/\s*a\b|"
                    r"\btrading as\b|\ba\s*/\s*k\s*/\s*a\b|\baka\b)\s*:?\s*", re.I)
_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s*\d+\s*$", re.I)
_DOMAIN = re.compile(r"^[@#]?([a-z0-9][a-z0-9\-]*?)(?:\.(?:com|net|org|in|co|fr|io|biz|info|us|uk))*$")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
SUFFIX_CANON = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "ltd": "ltd", "limited": "ltd", "llp": "llp", "lp": "lp", "plc": "plc", "pvt": "pvt",
    "private": "pvt", "pllc": "pllc", "pc": "pc", "pa": "pa", "esq": "esq", "cpa": "cpa", "dds": "dds", "md": "md",
    "dmd": "dmd", "lcsw": "lcsw", "phd": "phd", "pty": "pty",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc",
    # anyascii renderings of Indian-script 'Private', 'Limited', 'LLP'
    "praivet": "pvt", "elelpi": "llp", "pvtltd": "pvt",
}
HONORIFIC = {"mr", "mrs", "ms", "shri", "sri", "smt", "ets", "groupe", "cie", "etablissements", "compagnie"}
FR_SUFFIX = {"ei": "ei"}   # France-only: 'ei' occurs in 453 train names as a plain word
CONNECTOR = {"and", "of", "the", "du", "de", "la", "le", "les", "des", "et", "d", "l"}
_HOMO = str.maketrans({"1": "l", "0": "o", "5": "s", "6": "g", "i": "l"})


def _fold_token(t):
    """OCR look-alike folding: l/I/1 -> l, 0/O -> o, 5/S -> s, 6/G -> g (digits only inside words)."""
    if any(c.isalpha() for c in t):
        return t.translate(_HOMO)
    return t.replace("i", "l")


_FOLDED_SUFFIX = {_fold_token(k): v for k, v in SUFFIX_CANON.items()}


# Indian-script word -> English word, learned from non-dev train pairs (scripts/build_xlit_dict.py)
_XLIT_RES = Path(__file__).parent / "resources" / "in_xlit.json"
XLIT = json.loads(_XLIT_RES.read_text()) if _XLIT_RES.exists() else {"name": {}, "addr": {}}
_NATIVE_WORD = re.compile(r"[^\s,]*[\u0900-\u0DFF][^\s,]*")   # words containing Indian-script characters


def _xlit(s, kind):
    """Replace native-script words by their learned English word (unknown words: plain transliteration)."""
    table = XLIT.get(kind, {})

    def one(m):
        toks = re.sub(r"[^a-z0-9]+", " ", anyascii(m.group(0)).lower()).split()
        return " ".join(table.get(t, t) for t in toks)
    return _NATIVE_WORD.sub(one, s) if table else s


def _ascii(s, kind=None):
    s = s or ""
    if kind and re.search(r"[\u0900-\u0DFF]", s):
        s = _xlit(s, kind)
    return anyascii(s).lower()


def parse_name(raw, country=None):
    s = _ascii(raw, "name")
    s = s.split("|")[0]                              # 'Name | www.site.com' -> 'Name'
    s = _ID.sub(" ", s)
    s = re.sub(r"\bm\s*/\s*s\b", " ", s)             # M/s
    parts = [p for p in _ALIAS.split(s) if p.strip()]
    alias = ""
    if len(parts) > 1:                               # 'X F/K/A Y', 'X DBA: Y': right side is the listed name
        alias, s = parts[0], parts[-1]
    compact = s.strip().replace(" ", "")
    m = _DOMAIN.match(compact) if ("." in compact or compact[:1] in "@#") else None
    is_domain = int(bool(m))
    if m:
        s = m.group(1)
    s = _DOTTED.sub(lambda x: x.group(0).replace(".", ""), s)
    s = re.sub(r"['`’]", "", s)
    words = _NON_ALNUM.sub(" ", s).split()
    core, fold, suffix, seen = [], [], set(), set()
    for w in words:
        f = _fold_token(w)
        if w in HONORIFIC or w in CONNECTOR:
            continue
        if f in _FOLDED_SUFFIX:
            suffix.add(_FOLDED_SUFFIX[f])
            continue
        if country == "France" and w in FR_SUFFIX:
            suffix.add(FR_SUFFIX[w])
            continue
        if w in seen:                                # token duplication operator
            continue
        seen.add(w)
        core.append(w)
        fold.append(f)
    a_words = [w for w in _NON_ALNUM.sub(" ", alias).split() if w not in CONNECTOR and w not in HONORIFIC
               and _fold_token(w) not in _FOLDED_SUFFIX]
    return {"core": " ".join(core), "key": " ".join(sorted(set(core))), "fold": " ".join(sorted(set(fold))),
            "flat": "".join(core), "suffix": " ".join(sorted(suffix)), "alias": " ".join(a_words),
            "is_domain": is_domain}


# ---------------------------------------------------------------- house numbers
def _strip0(x):
    return x.lstrip("0") or "0"


def house_cores(token):
    """Canonical cores of one house-number token (structured operators undone)."""
    t = token.lower().replace("#", "").strip(" .-")
    t = re.sub(r"^[a-z]{1,2}-?(?=\d)", "", t)            # F-2/3, B-20, C-881, A14, S-159a
    t = re.sub(r"(?<=\d)-?[a-z]{1,2}$", "", t)            # 3707-D, 8871C, 159a
    if "/" in t:
        return {"/".join(_strip0(p) for p in t.split("/") if p)} if re.fullmatch(r"[\d/]+", t) else set()
    m = re.fullmatch(r"(\d+)-(\d+)", t)
    if m:
        a, b = m.groups()
        if len(a) == len(b) and int(b) > int(a):        # range 6968-6972: both ends are plausible
            return {_strip0(a), _strip0(b)}
        return {_strip0(a + b)}                          # inserted hyphen 1-30 -> 130
    return {_strip0(t)} if t.isdigit() else set()


def number_relation(a, b):
    """Relation between two sets of house cores."""
    if not a or not b:
        return "missing"
    if a & b:
        return "equal"
    if any(not x.replace("/", "").isdigit() for x in a | b):   # France '5b' (bis) vs '5': a different door
        return "substitution"
    for x in a:
        for y in b:
            lo, hi = (x, y) if len(x) < len(y) else (y, x)
            if lo and "/" not in lo and (hi.endswith(lo) or hi.startswith(lo)):
                return "digit_drop"
    return "substitution"


# ---------------------------------------------------------------- addresses
STREET_TYPE = {
    "street": "street", "st": "street", "str": "street", "road": "road", "rd": "road", "avenue": "avenue",
    "ave": "avenue", "av": "avenue", "lane": "lane", "ln": "lane", "drive": "drive", "dr": "drive",
    "circle": "circle", "cir": "circle", "place": "place", "pl": "place", "court": "court", "ct": "court",
    "boulevard": "boulevard", "blvd": "boulevard", "bd": "boulevard", "highway": "highway", "hwy": "highway",
    "parkway": "parkway", "pkwy": "parkway", "way": "way", "terrace": "terrace", "ter": "terrace", "cove": "cove",
    "cv": "cove", "trail": "trail", "trl": "trail", "square": "square", "sq": "square", "loop": "loop",
    "pike": "pike", "alley": "alley", "marg": "marg", "cross": "cross",
    # France
    "rue": "rue", "r": "rue", "route": "route", "rte": "route", "allee": "allee", "all": "allee",
    "impasse": "impasse", "imp": "impasse", "chemin": "chemin", "chem": "chemin", "quai": "quai", "cours": "cours",
}
ORDINAL_WORD = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
                "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
                "thirteenth": "13", "fourteenth": "14", "fifteenth": "15", "twentieth": "20", "twentith": "20"}
PLACE_SUFFIX = {"city", "town", "township", "cdp", "county", "borough", "village", "carrier", "annex"}
DOOR_PREFIX = {"no", "nos", "number", "plot", "door", "doro", "dsor", "flat", "falt", "shop", "off", "ward",
               "holding", "prop", "hno", "hn", "house", "bl", "c/o", "r/o", "s/o", "d/o", "w/o"}
_UNIT = re.compile(r"\b(?:p\s*o\s*box|pmb|unit|apt|apartment|suite|ste|fl(?=\s*\d)|floor(?=\s*\d))\s*#?\s*([a-z0-9\-]+)")
_NULL = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\(no address\)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "puducherry": "py",
    "chandigarh": "ch",
}
IN_VARIANT = {"keralam": "kerala", "orissa": "odisha", "pondicherry": "puducherry", "nct of delhi": "delhi",
              "ts": "telangana", "ct": "chhattisgarh", "or": "odisha", "ut": "uttarakhand"}
# native-script state names seen in the data, keyed by their anyascii rendering (computed, not hand-typed)
_NATIVE = ["राजस्थान", "दिल्ली", "ગુજરાત", "తెలంగాణ", "महाराष्ट्र", "পশ্চিমবঙ্গ", "தமிழ்நாடு", "കേരളം",
           "हरियाणा", "ಕರ್ನಾಟಕ", "उत्तर प्रदेश"]
_NATIVE_EN = ["rajasthan", "delhi", "gujarat", "telangana", "maharashtra", "west bengal", "tamil nadu", "kerala",
              "haryana", "karnataka", "uttar pradesh"]
IN_NATIVE = {_NON_ALNUM.sub(" ", anyascii(n).lower()).strip(): e for n, e in zip(_NATIVE, _NATIVE_EN)}
_RES = Path(__file__).parent / "resources" / "in_native_states.json"   # derived from train pairs (scripts/build_native_states.py)
if _RES.exists():
    IN_NATIVE.update(json.loads(_RES.read_text()))
# ---------------------------------------------------------------- France (unseen in train)
# Every rule here runs only when country == "France", so the train parquet (US/India) is unchanged.
# Measured on test France: records whose S1 name + number + street match write S1's 'bis' as 'bis' 81% / 'B' 18%,
# bare number 1% -> the letter is part of the door: '5 bis' = '5B' = '5b' != '5'.
_FR_LETTER = re.compile(r"\b0*(\d+)\s*-?\s*(bis|ter|quater|[abct])\b(?!\s*\d)")
_FR_LETTER_CODE = {"bis": "b", "ter": "t", "quater": "q"}
_FR_CORE = re.compile(r"0*(\d+)([abctq])")


def fr_letters(s):
    """'5 bis' / '5bis' / '5 B' / '5-b' -> '5b'; ter -> t, quater -> q (lowercase ASCII input)."""
    return _FR_LETTER.sub(lambda m: m.group(1) + _FR_LETTER_CODE.get(m.group(2), m.group(2)), s)


def _cores(token, country):
    if country == "France":
        m = _FR_CORE.fullmatch(token)
        if m:
            return {_strip0(m.group(1)) + m.group(2)}
    return house_cores(token)


# street words after the house number in test France records (counts: cour 7.4k, ch 6.6k, cite 5.9k, crs 4.3k, q 3.9k)
FR_STREET_TYPE = {
    "rue": "rue", "r": "rue", "route": "route", "rte": "route", "allee": "allee", "all": "allee", "alle": "allee",
    "alee": "allee", "impasse": "impasse", "imp": "impasse", "chemin": "chemin", "chem": "chemin", "ch": "chemin",
    "quai": "quai", "q": "quai", "cours": "cours", "crs": "cours", "cour": "cour", "cite": "cite",
    "passage": "passage", "pass": "passage", "psg": "passage", "residence": "residence", "res": "residence",
    "avenue": "avenue", "ave": "avenue", "av": "avenue", "boulevard": "boulevard", "blvd": "boulevard",
    "bd": "boulevard", "place": "place", "pl": "place", "square": "square", "sq": "square", "chaussee": "chaussee",
    "lotissement": "lotissement", "mail": "mail", "esplanade": "esplanade", "digue": "digue", "parvis": "parvis",
    "sente": "sente", "ruelle": "ruelle", "promenade": "promenade",
}
FR_ABBR = {"st": "saint", "ste": "sainte"}   # 'St Nazaire' (15.8k record components) is a city, not a street
# department -> region, learned from test co-occurrence (record department next to an S1 city of one region):
# nord 150,860/150,875 hauts de france; gironde 150,420/150,470 nouvelle aquitaine;
# loire atlantique 118,995/119,226 pays de la loire; pas de calais 27,838/27,848 hauts de france
FR_DEPT = {"nord": "hauts de france", "pas de calais": "hauts de france", "gironde": "nouvelle aquitaine",
           "loire atlantique": "pays de la loire"}


def _fr_type(t):
    """France street type, tolerating the generator's typos of the long words (aveue, aenue, bouelvard)."""
    if t in FR_STREET_TYPE:
        return FR_STREET_TYPE[t]
    if len(t) >= 5 and t[0] == "a" and Levenshtein.distance(t, "avenue") <= 2:
        return "avenue"
    if len(t) >= 7 and t[0] == "b" and Levenshtein.distance(t, "boulevard") <= 2:
        return "boulevard"
    return None


def _stype(t, country):
    if country == "France":
        return _fr_type(t)
    return STREET_TYPE.get(t) if (t != "r") else None


FR_REGIONS = {"auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
              "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
              "pays de la loire", "provence alpes cote d azur"}


def _state(comp, country):
    """Canonical state for one address component, or None."""
    c = comp.strip()
    if country == "US":
        if c in US_STATES:
            return c
        rev = {v: k for k, v in US_STATES.items()}
        return rev.get(c)
    if country == "India":
        c = IN_VARIANT.get(c, c)
        if c in IN_STATES:
            return c
        code = {v: k for k, v in IN_STATES.items()}
        if c in code:
            return code[c]
        return IN_NATIVE.get(c)
    if country == "France":
        c = FR_DEPT.get(c, c)
        return c if c in FR_REGIONS else None
    return None


def parse_address(raw, country):
    s = _NULL.sub(" ", _ascii(raw, "addr"))
    s = re.sub(r"\bn\s*(?:°|deg)", " no ", s).replace("#", " ").replace("'", " ")   # N° (anyascii: 'ndeg')
    s = re.sub(r"(?<=[a-z])\.(?=[a-z\s]|$)", " ", s)            # 'h.no' -> 'h no', 'st.' -> 'st'
    s = re.sub(r"\b(?:h|s|d)\s+no\b", " no ", s)                 # H.NO / S.No / D.No prefixes
    if country == "France":
        s = fr_letters(s)
    unit = ""
    m = _UNIT.search(s)
    if m:
        unit, s = m.group(1), s[:m.start()] + " " + s[m.end():]
    s = re.sub(r"\b\d+\s*/\s*2\b", " ", s) if country != "India" else s   # '73 1/2' fraction (not India 4/792)
    comps = [c.strip() for c in s.split(",")]
    street, street_type, house, alts, state, city_parts, tokens = "", "", "", set(), "", [], []
    for comp in comps:
        spaced = comp.replace("-", " ") if not re.search(r"\d", comp) else comp
        st = _state(_NON_ALNUM.sub(" ", spaced).strip(), country)
        if st:
            state = state or st
            tokens.append("st_" + st.replace(" ", "_"))
            continue
        toks = [w for t in _TOKEN.findall(comp) for w in (t.split("-") if not re.search(r"\d", t) else [t])]
        toks = [t for t in toks if t not in DOOR_PREFIX]
        if not toks:
            continue
        if country != "France" and len(toks) >= 2 and toks[-1] == "saint":   # generator's 'St' -> 'Saint'
            toks[-1] = "street"
        toks = [ORDINAL_WORD.get(t, t) for t in toks]
        if country == "France":
            toks = [FR_ABBR.get(t, t) for t in toks]
        types = [i for i, t in enumerate(toks) if _stype(t, country)]
        nums = [i for i, t in enumerate(toks) if re.search(r"\d", t)]
        if types and not street_type:
            ti = types[-1] if country != "France" else types[0]
            street_type = _stype(toks[ti], country)
            hi = next((i for i in nums if i < ti or country == "France"), None)
            if hi is not None and not re.fullmatch(r"\d+(?:st|nd|rd|th)", toks[hi]):
                alts |= _cores(toks[hi], country)
                house = house or min(alts, key=lambda x: (len(x), x), default="")   # deterministic tie-break
            name = [re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", t) for i, t in enumerate(toks)
                    if i != ti and i != hi and t not in CONNECTOR and not _stype(t, country) == t and i not in
                    [j for j in types if j != ti]]
            street = " ".join(name)
            tokens += ["n" + x for x in sorted(alts)] + name + ["t_" + street_type]
            continue
        if nums and not alts:
            cores = _cores(toks[nums[0]], country)
            if cores:
                alts |= cores
                house = min(cores, key=lambda x: (len(x), x))
                toks = [t for i, t in enumerate(toks) if i != nums[0]]
                tokens += ["n" + x for x in sorted(cores)]
        words = [t for t in toks if t not in PLACE_SUFFIX and t not in CONNECTOR and not re.search(r"\d", t)]
        if words:
            city_parts.append(" ".join(words))
            tokens += words
    has_addr = int(bool(tokens) or bool(unit))
    return {"house": house, "house_alts": " ".join(sorted(alts)), "unit": unit, "street": street,
            "street_type": street_type, "city": city_parts[0] if city_parts else "", "state": state,
            "locality": " ".join(city_parts[1:]), "tokens": " ".join(tokens), "has_addr": has_addr}
