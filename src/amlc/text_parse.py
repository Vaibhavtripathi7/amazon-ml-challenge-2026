"""Pull structured fields out of 2025-style catalog_content (Item Name / Bullet Point / Value / Unit)."""
import re

import numpy as np
import pandas as pd

# spelling (lowercase, dots removed, trailing s stripped) -> standard unit
_UNITS = {
    "ounce": "oz", "oz": "oz",
    "fl oz": "fl_oz", "fluid ounce": "fl_oz", "fl ounce": "fl_oz", "fluid oz": "fl_oz",
    "count": "count", "ct": "count", "each": "count", "pack": "count", "piece": "count", "unit": "count",
    "pound": "lb", "lb": "lb", "gram": "g", "g": "g", "gm": "g", "kg": "kg", "kilogram": "kg",
    "ml": "ml", "millilitre": "ml", "milliliter": "ml", "liter": "l", "litre": "l", "l": "l",
    "none": "none",
}
_TO_OZ = {"oz": 1.0, "lb": 16.0, "g": 0.035274, "kg": 35.274, "fl_oz": 1.0, "ml": 0.033814, "l": 33.814}


def _norm_unit(u):
    if not isinstance(u, str):
        return "none"
    u = re.sub(r"[.]", "", u.strip().lower())
    u = re.sub(r"\s+", " ", u)
    return _UNITS.get(u) or _UNITS.get(re.sub(r"s$", "", u)) or _UNITS.get(re.sub(r"es$", "", u)) or "other"


def _field(s, label):
    return s.str.extract(rf"^{label}: (.*)$", flags=re.M)[0].str.strip()


def parse_catalog(text):
    """text: Series of catalog_content. Returns a DataFrame of numeric/categorical features, same index."""
    text = text.fillna("")
    value = pd.to_numeric(_field(text, "Value"), errors="coerce")
    unit = _field(text, "Unit").map(_norm_unit)
    name = _field(text, "Item Name").fillna("")
    pack = pd.to_numeric(text.str.extract(r"[Pp]ack of (\d+)")[0], errors="coerce").fillna(1)
    return pd.DataFrame({
        "value": value,
        "unit": unit,
        "value_oz": value * unit.map(_TO_OZ),
        "pack": pack,
        "log_value": np.log1p(value),
        "name_len": name.str.len(),
        "text_len": text.str.len(),
        "n_bullets": text.str.count(r"(?m)^Bullet Point"),
        "has_desc": text.str.contains(r"(?m)^Product Description:").astype(int),
    }, index=text.index)
