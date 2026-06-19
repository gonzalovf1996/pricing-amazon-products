import re
import pandas as pd


UNIT_MAP = {
    "oz": "ounce",
    "ounces": "ounce",
    "ounce": "ounce",
    "fl oz": "fluid_ounce",
    "fl. oz": "fluid_ounce",
    "fluid ounce": "fluid_ounce",
    "fluid ounces": "fluid_ounce",
    "fluid ounce(s)": "fluid_ounce",
    "lb": "pound",
    "pound": "pound",
    "ct": "count",
    "count": "count",
    "each": "count"
}

def extract_catalog_subsections(text: str) -> list[str]:
    labels = []
    for line in str(text).splitlines():
        match = re.match(r"^\s*([^:]+)\s*:", line)
        if match:
            label = match.group(1).strip()
            labels.append(label)
    return labels


def extract_catalog_field(text: str, field_name: str) -> str:
    if pd.isna(text):
        return pd.NA

    text = str(text)

    for line in text.splitlines():
        line = line.strip()
        
        if line.startswith(f"{field_name}"):
            value = line.split(":", 1)[1].strip()
            return value if value else pd.NA

    return pd.NA