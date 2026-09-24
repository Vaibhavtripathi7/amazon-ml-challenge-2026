import numpy as np
import pandas as pd

from amlc.text_parse import parse_catalog


def one(text):
    return parse_catalog(pd.Series([text])).iloc[0]


def test_value_unit_and_pack_are_extracted():
    r = one("Item Name: Taco Sauce, 12 Ounce (Pack of 6)\nValue: 72.0\nUnit: Fl Oz\n")
    assert r.value == 72.0 and r.unit == "fl_oz" and r.pack == 6


def test_unit_spellings_are_normalised():
    for spelling in ["Ounce", "oz", "OUNCES", "ounce "]:
        assert one(f"Item Name: x\nValue: 1\nUnit: {spelling}\n").unit == "oz"
    for spelling in ["Fl Oz", "fl. oz", "Fluid Ounce", "fluid ounces"]:
        assert one(f"Item Name: x\nValue: 1\nUnit: {spelling}\n").unit == "fl_oz"
    for spelling in ["Count", "ct", "each", "packs"]:
        assert one(f"Item Name: x\nValue: 1\nUnit: {spelling}\n").unit == "count"


def test_weight_units_convert_to_ounces():
    assert one("Item Name: x\nValue: 2\nUnit: Pound\n").value_oz == 32.0
    assert abs(one("Item Name: x\nValue: 100\nUnit: gram\n").value_oz - 3.5274) < 1e-3


def test_missing_value_is_nan_not_crash():
    r = one("Item Name: x\nValue: nan\nUnit: None\n")
    assert np.isnan(r.value) and r.unit == "none" and r.pack == 1
