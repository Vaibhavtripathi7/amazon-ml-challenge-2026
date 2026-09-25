"""Text cleaning for business names and addresses. Country-agnostic on purpose (France is unseen in train)."""
import re
import unicodedata

LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llp", "lp",
    "pvt", "private", "plc", "the",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc",
}
_NON_WORD = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d")


def _fold(s):
    """Lowercase and strip accents (é -> e); non-Latin scripts are kept as-is."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def name_key(name):
    """Order-free key: cleaned words minus legal/filler words, sorted. 'L.L.C.' -> 'llc' -> dropped."""
    words = _NON_WORD.sub("", _fold(name).replace("&", " ")).split()
    return " ".join(sorted(w for w in words if w not in LEGAL))


def house_number(address):
    """Digits of the first token that contains a digit ('B-#20' -> '20'); '' if none."""
    for tok in re.split(r"[\s,]+", address):
        if _DIGITS.search(tok):
            return "".join(re.findall(r"\d", tok))
    return ""
