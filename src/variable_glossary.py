"""
variable_glossary.py
Map coded variable names (l_CLI38, u_TOP109, etc.) to human-readable
ecological descriptions using the S2 Variable Metadata Glossary.

Usage:
    from variable_glossary import load_glossary, translate
    glossary = load_glossary("data/raw/S2.xlsx")
    print(translate("l_CLI38", glossary))
    # -> "Mean temp. warmest quarter [mean] (°C)"
"""

from pathlib import Path
import pandas as pd


def load_glossary(filepath: str = "data/raw/S2.xlsx") -> dict:
    """
    Parse the S2 glossary XLSX and return a dict mapping variable codes
    to short human-readable descriptions.

    Returns dict like {"l_CLI38": "Mean temp. warmest quarter [mean] (°C)", ...}
    """
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: Glossary file not found at {filepath}")
        return {}

    glossary = {}

    # --- CLIMATE ---
    cli = pd.read_excel(filepath, sheet_name="CLIMATE")
    for _, row in cli.iterrows():
        local = _clean_code(row.get("Local Climate", ""))
        upstream = _clean_code(row.get("Upstream Climate", ""))
        definition = str(row.get("Definition", ""))
        unit = str(row.get("Unit", ""))
        short = _shorten_definition(definition, unit)
        if local:
            glossary[local] = short
        if upstream:
            glossary[upstream] = short

    # --- SOIL ---
    sol = pd.read_excel(filepath, sheet_name="SOIL")
    for _, row in sol.iterrows():
        local = _clean_code(row.get("Local Soil", ""))
        upstream = _clean_code(row.get("Upstream Soil", ""))
        definition = str(row.get("Definition", ""))
        unit = str(row.get("Unit", ""))
        short = _shorten_definition(definition, unit)
        if local:
            glossary[local] = short
        if upstream:
            glossary[upstream] = short

    # --- LAND COVER ---
    lac = pd.read_excel(filepath, sheet_name="LAND COVER")
    for _, row in lac.iterrows():
        local = _clean_code(row.get("Local Climate", ""))
        upstream = _clean_code(row.get("Upstream Climate", ""))
        definition = str(row.get("Definition", ""))
        unit = str(row.get("Unit", ""))
        short = _shorten_definition(definition, unit)
        if local:
            glossary[local] = short
        if upstream:
            glossary[upstream] = short

    # --- TOPOGRAPHY ---
    top = pd.read_excel(filepath, sheet_name="TOPOGRAPHY")
    for _, row in top.iterrows():
        local = _clean_code(row.get("Local Topography", ""))
        upstream = _clean_code(row.get("Upstream Topography", ""))
        definition = str(row.get("Definition (Hydrography90m)", ""))
        unit = str(row.get("Unit", ""))
        stat = _extract_stat(definition)
        short = _shorten_definition(definition, unit)
        if local:
            glossary[local] = short
        if upstream:
            glossary[upstream] = short

    return glossary


def translate(code: str, glossary: dict) -> str:
    """Translate a variable code to its description, or return the code."""
    return glossary.get(code, code)


def translate_list(codes: list, glossary: dict) -> list:
    """Translate a list of variable codes."""
    return [translate(c, glossary) for c in codes]


def make_label(code: str, glossary: dict, max_len: int = 50) -> str:
    """
    Make a short label for plots: "code: description" truncated to max_len.
    """
    desc = glossary.get(code, "")
    if not desc:
        return code
    label = f"{code}: {desc}"
    if len(label) > max_len:
        label = label[:max_len - 3] + "..."
    return label


def _clean_code(raw) -> str:
    """Clean a code like 'l-CLI38' to 'l_CLI38'."""
    s = str(raw).strip()
    if s == "nan" or not s:
        return ""
    return s.replace("-", "_")


def _extract_stat(definition: str) -> str:
    """Extract the statistic type [minimum], [maximum], [mean], [sd]."""
    for stat in ["minimum", "maximum", "mean", "sd"]:
        if f"[{stat}]" in definition.lower():
            return stat
    return ""


def _shorten_definition(definition: str, unit: str) -> str:
    """
    Shorten a long definition to something usable in plots.

    'Annual mean air temperature [maximum]' -> 'Ann. mean air temp. [max] (°C)'
    """
    s = definition.strip()

    # Shorten stat brackets
    s = s.replace("[minimum]", "[min]")
    s = s.replace("[maximum]", "[max]")
    s = s.replace("[mean]", "[mean]")

    # Common abbreviations
    replacements = {
        "temperature": "temp.",
        "Temperature": "Temp.",
        "precipitation": "precip.",
        "Precipitation": "Precip.",
        "Annual": "Ann.",
        "annual": "ann.",
        "coefficient of variation": "CV",
        "standard deviation": "SD",
        "volumetric fraction": "vol. frac.",
        "Elevation difference between focal grid cell and": "Elev. diff. to",
        "Distance between focal grid cell and": "Dist. to",
        "between focal cell and downstream cell": "focal-downstream",
        "between highest upstream cell, focal cell, and downstream cell": "up-focal-down",
        "between lowest upstream cell, focal cell, and downstream cell": "low up-focal-down",
        "the outlet grid cell in the network": "basin outlet",
        "the downstream stream node grid cell": "downstream node",
        "(closed to open >15%)": "",
        "(codes 10, 11, 12)": "",
        "(60, 61, 62)": "",
        "(70, 71, 72)": "",
        "(80, 81, 82)": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Add unit
    unit = str(unit).strip()
    if unit and unit != "nan":
        s = f"{s} ({unit})"

    return s.strip()


def print_glossary_for_variables(
    variables: list, glossary: dict
) -> None:
    """Print a formatted table of variable codes and their meanings."""
    print(f"\n{'Code':<15s} {'Description'}")
    print("-" * 70)
    for var in variables:
        desc = glossary.get(var, "UNKNOWN")
        print(f"{var:<15s} {desc}")


# --- Standalone usage ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Variable glossary lookup")
    parser.add_argument("--glossary", "-g", default="data/raw/S2.xlsx",
                        help="Path to S2 glossary XLSX")
    parser.add_argument("--variables", "-v", nargs="*",
                        help="Variable codes to look up")
    args = parser.parse_args()

    glossary = load_glossary(args.glossary)
    print(f"Loaded {len(glossary)} variable definitions")

    if args.variables:
        print_glossary_for_variables(args.variables, glossary)
    else:
        # Print all
        print_glossary_for_variables(sorted(glossary.keys()), glossary)