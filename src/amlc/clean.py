"""Cleaning for names and addresses. Country-agnostic (France is unseen in train).

clean_name:    transliterate any script -> lowercase ASCII, drop punctuation/junk and legal words, keep word order.
clean_address: same folding, standard street words, numbers pulled out separately (leading zeros and
               ordinal suffixes removed) so '001300' == '1300' and '152ND' == '152'.
"""
import re

from anyascii import anyascii

LEGAL = {
    # English / India
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llp", "lp",
    "pvt", "private", "plc", "the", "and",
    # transliterated Indian-script spellings of the same words (anyascii output)
    "praivet", "limited", "elelpi", "pvtltd",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc",
}
STREET = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "cir": "circle", "pl": "place", "ct": "court", "blvd": "boulevard", "bd": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "apt": "apartment", "fl": "floor", "sq": "square",
}
DROP_ADDR = {"null", "none", "na", "no", "hno", "hn", "c", "o", "r"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"^[a-z]?0*(\d+)[a-z]{0,2}$")  # 152nd, 114b, a14, 001300
_FR_NUM = re.compile(r"0*(\d+)([abctq])")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?")  # l.l.c. / p.v.t


def _fold(s):
    s = anyascii(s).lower()
    return _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)


def clean_name(name):
    words = _NON_ALNUM.sub(" ", _fold(name)).split()
    return " ".join(w for w in words if w not in LEGAL)


def clean_address(address, country=None):
    """Returns (words, numbers) as space-joined strings. France keeps bis/ter as part of the number ('5b')."""
    words, nums = [], []
    text = _fold(address)
    if country == "France":
        from amlc.norm import fr_letters
        text = fr_letters(text)
    for tok in _NON_ALNUM.sub(" ", text).split():
        m = _NUMBER.match(tok)
        fr = _FR_NUM.fullmatch(tok) if country == "France" else None
        if fr:
            nums.append((fr.group(1).lstrip("0") or "0") + fr.group(2))
        elif m:
            nums.append(m.group(1) or "0")
        elif tok not in DROP_ADDR:
            words.append(STREET.get(tok, tok))
    return " ".join(words), " ".join(nums)
