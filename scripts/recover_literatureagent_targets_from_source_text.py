from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from pathlib import Path

import pandas as pd


AGGREGATE_NAMES = {
    "all_records.csv",
    "all_records_from_manager.csv",
    "paper_summaries_from_manager.csv",
}


def present(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "[]", "{}", "not reported", "unknown", "n/a"}


def as_float(value) -> float | None:
    try:
        out = float(str(value).strip())
    except Exception:
        return None
    return out if out == out else None


def model_range_pce(value) -> bool:
    val = as_float(value)
    return val is not None and 5 <= val <= 30


PCE_PATTERNS = [
    (
        "explicit_pce_after",
        re.compile(
            r"\b(?:PCE|power conversion efficienc(?:y|ies)|photo[- ]?conversion efficienc(?:y|ies)|photoconversion efficienc(?:y|ies))"
            r"\s*(?:of|=|:|reached|reaches|achieved|achieves|delivered|yields|yielded|up to|exceeding|over|above|as high as|was|is)?"
            r"\s*(\d{1,2}(?:\.\d+)?)\s*%",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "explicit_pce_before",
        re.compile(
            r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:PCE|power conversion efficiency|photo[- ]?conversion efficiency|photoconversion efficiency)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "device_efficiency",
        re.compile(
            r"\b(?:champion|best|maximum|optimized|control|target|certified|record|device|cell|PSC|solar cell)[^.\n]{0,140}"
            r"\b(?:efficienc(?:y|ies)|performance)[^.\n]{0,90}?(\d{1,2}(?:\.\d+)?)\s*%",
            flags=re.IGNORECASE,
        ),
    ),
]

VOC_PATTERNS = [
    re.compile(r"\bV(?:oc|OC)\s*(?:=|:|of|was|is)?\s*(\d(?:\.\d+)?)\s*V\b", flags=re.IGNORECASE),
    re.compile(r"\bopen[- ]circuit voltage\s*(?:=|:|of|was|is)?\s*(\d(?:\.\d+)?)\s*V\b", flags=re.IGNORECASE),
]

JSC_PATTERNS = [
    re.compile(r"\bJ(?:sc|SC)\s*(?:=|:|of|was|is)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*cm\s*(?:-2|-?2|\^-2)", flags=re.IGNORECASE),
    re.compile(r"\bshort[- ]circuit current(?: density)?\s*(?:=|:|of|was|is)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*cm\s*(?:-2|-?2|\^-2)", flags=re.IGNORECASE),
]

FF_PATTERNS = [
    re.compile(r"\bFF\s*(?:=|:|of|was|is)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?", flags=re.IGNORECASE),
    re.compile(r"\bfill factor\s*(?:=|:|of|was|is)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?", flags=re.IGNORECASE),
]

FORMULA_PATTERNS = [
    re.compile(r"\b(?:FAPbI3|MAPbI3|CsPbI3|CsPbBr3|MASnI3|FASnI3|CH3NH3PbI3|CH3NH3SnI3)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:CH3NH3|MA|FA|Cs|Rb|K)(?:Pb|Sn|Ge)(?:I|Br|Cl)3\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:(?:CH3NH3|MA|FA|Cs|Rb|K)(?:\d(?:\.\d+)?)?){1,5}(?:Pb|Sn|Ge)(?:I|Br|Cl)(?:\d(?:\.\d+)?)?(?:(?:I|Br|Cl)(?:\d(?:\.\d+)?)?){0,2}\b", flags=re.IGNORECASE),
]

GOOD_CONTEXT = re.compile(
    r"\b(PCE|power conversion|photoconversion|device|cell|solar|photovoltaic|PSC|champion|certified|Voc|Jsc|fill factor|FF|J-V|JV)\b",
    flags=re.IGNORECASE,
)

BAD_CONTEXT = re.compile(
    r"\b(retention|retained|humidity|RH|relative humidity|degradation|quantum yield|PLQY|EQE|IPCE|transmittance|absorbance|external quantum efficiency)\b",
    flags=re.IGNORECASE,
)

STABILITY_CONTEXT = re.compile(
    r"\b(stability|stable|aging|ageing|aged|lifetime|T80|T95|retention|retained|maintained|degradation|MPP|maximum power point|ISOS|humidity|RH|thermal|illumination|encapsulat|unencapsulat)\b",
    flags=re.IGNORECASE,
)

T80_PATTERNS = [
    re.compile(r"\bT\s*80\b\s*(?:=|:|of|was|is|reached|exceeded|over|above|>|~|approximately|about)?\s*(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b", flags=re.IGNORECASE),
    re.compile(r"\b(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b[^.\n]{0,80}?\bT\s*80\b", flags=re.IGNORECASE),
]

T95_PATTERNS = [
    re.compile(r"\bT\s*95\b\s*(?:=|:|of|was|is|reached|exceeded|over|above|>|~|approximately|about)?\s*(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b", flags=re.IGNORECASE),
    re.compile(r"\b(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b[^.\n]{0,80}?\bT\s*95\b", flags=re.IGNORECASE),
]

RETENTION_PATTERNS = [
    re.compile(
        r"\b(?:retained|retention|maintained|remaining|retains?)\b[^.\n]{0,120}?"
        r"(\d{2,3}(?:\.\d+)?)\s*%[^.\n]{0,140}?"
        r"(?:after|for|over|during|under|at)?\s*(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{2,3}(?:\.\d+)?)\s*%[^.\n]{0,120}?"
        r"\b(?:retained|retention|maintained|remaining)\b[^.\n]{0,140}?"
        r"(?:after|for|over|during|under|at)?\s*(\d{1,5}(?:[.,]\d+)?)\s*(?:h|hours|hrs)\b",
        flags=re.IGNORECASE,
    ),
]

AFTER_1000H_PATTERNS = [
    re.compile(
        r"\b(?:retained|maintained|remaining|preserved|retains?)\b[^.\n]{0,60}?"
        r"(\d{2,3}(?:\.\d+)?)\s*%[^.\n]{0,60}?"
        r"\b(?:after|for|over)\s*(?:1000|1,000)\s*(?:h|hours|hrs)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:after|for|over)\s*(?:1000|1,000)\s*(?:h|hours|hrs)\b[^.\n]{0,80}?"
        r"\b(?:retained|maintained|remaining|preserved|retains?)\b[^.\n]{0,60}?"
        r"(\d{2,3}(?:\.\d+)?)\s*%",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{2,3}(?:\.\d+)?)\s*%\s*(?:of (?:the )?initial)?\s*(?:PCE|efficiency|performance)?"
        r"[^.\n]{0,30}?\b(?:after|for|over)\s*(?:1000|1,000)\s*(?:h|hours|hrs)\b",
        flags=re.IGNORECASE,
    ),
]

SOLVENT_TERMS = [
    "DMF", "DMSO", "NMP", "GBL", "CB", "chlorobenzene", "toluene",
    "IPA", "isopropanol", "ethanol", "methanol", "acetonitrile",
    "ethyl acetate", "diethyl ether", "anisole",
]

ADDITIVE_TERMS = [
    "MACl", "FACl", "NH4Cl", "Pb(SCN)2", "KSCN", "KI", "KCl", "RbI", "CsI",
    "PEAI", "BAI", "FPEAI", "OAI", "PEACl", "EDAI2", "BDAI2", "GABr",
    "MAAc", "FABr", "MABr", "PbI2", "PbBr2",
]

PROCESS_TERMS = [
    ("spin coating", r"\bspin[- ]coat(?:ing|ed)?\b"),
    ("blade coating", r"\bblade[- ]coat(?:ing|ed)?\b"),
    ("slot-die coating", r"\bslot[- ]die\b"),
    ("thermal evaporation", r"\bthermal(?:ly)? evaporat(?:ion|ed)\b"),
    ("spray coating", r"\bspray[- ]coat(?:ing|ed)?\b"),
    ("doctor blading", r"\bdoctor blad(?:ing|ed)\b"),
    ("chemical bath deposition", r"\bchemical bath deposition\b|\bCBD\b"),
    ("atomic layer deposition", r"\batomic layer deposition\b|\bALD\b"),
    ("sputtering", r"\bsputter(?:ing|ed)?\b"),
]

SHAP_PRIORITY_FIELDS = [
    "Cell_area_measured",
    "ETL_thickness",
    "HTL_thickness_list",
    "Backcontact_thickness_list",
    "ETL_deposition_procedure",
    "HTL_deposition_procedure",
    "Backcontact_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_solvents_mixing_ratios",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_additives_compounds",
    "ETL_additives_compounds",
    "HTL_additives_compounds",
]

TEMP_PATTERNS = [
    re.compile(r"\b(?:at|under|temperature|thermal|heated|aging|ageing)[^.\n]{0,80}?(\d{1,3}(?:\.\d+)?)\s*(?:°\s*C|deg(?:ree)?s?\s*C|C)\b", flags=re.IGNORECASE),
]

RH_PATTERNS = [
    re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:RH|relative humidity|humidity)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:RH|relative humidity|humidity)\s*(?:of|=|:|at)?\s*(\d{1,3}(?:\.\d+)?)\s*%", flags=re.IGNORECASE),
]

LIGHT_PATTERNS = [
    re.compile(r"\b(1(?:\.0)?|0?\.\d+|\d+(?:\.\d+)?)\s*(?:sun|suns)\b", flags=re.IGNORECASE),
    re.compile(r"\b(\d{2,4}(?:\.\d+)?)\s*mW\s*cm\s*(?:-2|-?2|\^-2)\b", flags=re.IGNORECASE),
]

REFERENCE_CONTEXT = re.compile(
    r"\b(references|bibliography|copyright|publisher|permission|adapted with permission|all rights reserved|reported by others|previously reported)\b",
    flags=re.IGNORECASE,
)


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_validation_pass_keys(path: Path | None) -> set[tuple[str, int]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Validation report not found: {path}")
    df = read_csv(path)
    required = {"paper_slug", "row_index", "validation_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Validation report missing required columns: {sorted(missing)}")
    passed = df[df["validation_status"].astype(str).str.lower() == "pass"]
    out = set()
    for _, row in passed.iterrows():
        try:
            out.add((str(row["paper_slug"]), int(row["row_index"])))
        except Exception:
            continue
    return out


def first_numeric(text: str, patterns: list[re.Pattern], lo: float, hi: float) -> float | None:
    for pattern in patterns:
        m = pattern.search(text)
        if not m:
            continue
        val = as_float(m.group(1))
        if val is not None and lo <= val <= hi:
            return val
    return None


def candidate_score(kind: str, value: float, window: str, match_text: str) -> int:
    score = 0
    if "PCE" in match_text or re.search(r"power conversion|photoconversion", match_text, flags=re.IGNORECASE):
        score += 4
    if kind.startswith("explicit_pce"):
        score += 3
    if re.search(r"\b(champion|best|optimized|certified|record|device|cell|PSC|solar cell|J-V|JV)\b", window, flags=re.IGNORECASE):
        score += 3
    if first_numeric(window, VOC_PATTERNS, 0.4, 1.5) is not None:
        score += 1
    if first_numeric(window, JSC_PATTERNS, 1.0, 35.0) is not None:
        score += 1
    if first_numeric(window, FF_PATTERNS, 0.2, 90.0) is not None:
        score += 1
    if BAD_CONTEXT.search(window) and not re.search(r"\b(PCE|power conversion|device|cell|PSC|solar)\b", match_text, flags=re.IGNORECASE):
        score -= 5
    if REFERENCE_CONTEXT.search(window):
        score -= 3
    if value < 8 or value > 28:
        score -= 1
    return score


def extract_best_performance(text: str, min_score: int = 9) -> dict:
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    candidates = []
    for kind, pattern in PCE_PATTERNS:
        for match in pattern.finditer(cleaned):
            value = as_float(match.group(1))
            if value is None or not (5 <= value <= 30):
                continue
            start = max(0, match.start() - 220)
            end = min(len(cleaned), match.end() + 220)
            window = cleaned[start:end]
            if not GOOD_CONTEXT.search(window):
                continue
            score = candidate_score(kind, value, window, match.group(0))
            if score < min_score:
                continue
            candidates.append({
                "value": value,
                "score": score,
                "kind": kind,
                "snippet": window.strip(),
                "voc": first_numeric(window, VOC_PATTERNS, 0.4, 1.5),
                "jsc": first_numeric(window, JSC_PATTERNS, 1.0, 35.0),
                "ff": first_numeric(window, FF_PATTERNS, 0.2, 90.0),
            })
    if not candidates:
        return {}
    return sorted(candidates, key=lambda d: (d["score"], d["value"]), reverse=True)[0]


def normalize_number(value: str) -> float | None:
    return as_float(str(value).replace(",", ""))


def stability_score(window: str) -> int:
    score = 0
    if STABILITY_CONTEXT.search(window):
        score += 4
    if re.search(r"\b(device|cell|PSC|solar cell|PCE|efficiency|perovskite)\b", window, flags=re.IGNORECASE):
        score += 2
    if re.search(r"\b(unencapsulated|encapsulated|ambient|air|nitrogen|RH|humidity|illumination|MPP|maximum power point|ISOS|thermal)\b", window, flags=re.IGNORECASE):
        score += 2
    if REFERENCE_CONTEXT.search(window):
        score -= 4
    if re.search(r"\b(review|perspective|reported by others|literature)\b", window, flags=re.IGNORECASE):
        score -= 2
    return score


def best_stability_numeric(text: str, patterns: list[re.Pattern], lo: float, hi: float, min_score: int = 6) -> dict:
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    candidates = []
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            value = normalize_number(match.group(1))
            if value is None or not (lo <= value <= hi):
                continue
            start = max(0, match.start() - 240)
            end = min(len(cleaned), match.end() + 240)
            window = cleaned[start:end]
            score = stability_score(window)
            if score < min_score:
                continue
            candidates.append({"value": value, "score": score, "snippet": window.strip()})
    if not candidates:
        return {}
    return sorted(candidates, key=lambda d: (d["score"], d["value"]), reverse=True)[0]


def extract_retention(text: str, min_score: int = 6) -> dict:
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    candidates = []
    for pattern in RETENTION_PATTERNS:
        for match in pattern.finditer(cleaned):
            pct = normalize_number(match.group(1))
            hours = normalize_number(match.group(2))
            if pct is None or hours is None or not (1 <= pct <= 120) or not (1 <= hours <= 100000):
                continue
            pct_context = cleaned[max(0, match.start(1) - 20):min(len(cleaned), match.end(1) + 35)]
            if re.search(r"\b(RH|relative humidity|humidity)\b", pct_context, flags=re.IGNORECASE):
                continue
            start = max(0, match.start() - 240)
            end = min(len(cleaned), match.end() + 240)
            window = cleaned[start:end]
            local_prefix = cleaned[max(0, match.start() - 60):match.start()]
            if re.search(r"\b(loss|lost|lose|loses|degrad(?:e|es|ed|ation) by|drop(?:s|ped)? by|decrease(?:s|d)? by|decay(?:s|ed)? by)\b", local_prefix, flags=re.IGNORECASE):
                continue
            score = stability_score(window)
            if hours >= 100:
                score += 1
            if hours >= 500:
                score += 1
            if score < min_score:
                continue
            candidates.append({"retention_pct": pct, "hours": hours, "score": score, "snippet": window.strip()})
    if not candidates:
        return {}
    return sorted(candidates, key=lambda d: (d["score"], d["hours"], d["retention_pct"]), reverse=True)[0]


def extract_after_1000h(text: str, min_score: int = 6) -> dict:
    # Disabled as an independent parser. In practice, "1000 h" appears in
    # review tables, adjacent references, and multi-condition paragraphs where
    # it is too easy to bind the wrong percentage to the 1000 h condition.
    # Stability_PCE_after_1000_h is filled only from the explicit retention
    # parser when the same retention sentence reports ~1000 h exposure.
    return {}


def extract_stability_conditions(text: str) -> dict:
    out = {}
    cleaned = re.sub(r"\s+", " ", str(text or ""))
    temp = first_numeric(cleaned, TEMP_PATTERNS, 0, 120)
    rh = first_numeric(cleaned, RH_PATTERNS, 0, 100)
    light = first_numeric(cleaned, LIGHT_PATTERNS, 0.01, 1000)
    if temp is not None:
        out["temperature_c"] = temp
    if rh is not None:
        out["rh_percent"] = rh
    if light is not None:
        out["light"] = light / 100.0 if light > 10 else light
    atmosphere_terms = find_terms(cleaned, ["ambient", "air", "nitrogen", "N2", "argon", "glovebox"])
    if atmosphere_terms:
        out["atmosphere"] = atmosphere_terms
    if re.search(r"\bunencapsulated\b", cleaned, flags=re.IGNORECASE):
        out["encapsulation"] = "False"
    elif re.search(r"\bencapsulated\b", cleaned, flags=re.IGNORECASE):
        out["encapsulation"] = "True"
    return out


def extract_best_stability(text: str) -> dict:
    t80 = best_stability_numeric(text, T80_PATTERNS, 1, 100000)
    t95 = best_stability_numeric(text, T95_PATTERNS, 1, 100000)
    retention = extract_retention(text)
    after_1000h = extract_after_1000h(text)
    condition_text = "\n".join(
        x.get("snippet", "")
        for x in [t80, t95, retention, after_1000h]
        if x
    ) or text[:12000]
    return {
        "t80": t80,
        "t95": t95,
        "retention": retention,
        "after_1000h": after_1000h,
        "conditions": extract_stability_conditions(condition_text),
    }


def extract_formula(text: str) -> str | None:
    for pattern in FORMULA_PATTERNS:
        m = pattern.search(text)
        if m:
            formula = m.group(0).strip(" /,;:()[]{}")
            if any(x in formula for x in ["Pb", "Sn", "Ge"]) and any(x in formula for x in ["I", "Br", "Cl"]):
                return formula
    return None


def find_terms(text: str, terms: list[str]) -> str | None:
    hits = []
    for term in terms:
        pat = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(term)
    return "; ".join(dict.fromkeys(hits)) if hits else None


def first_sentence_with(text: str, pattern: str, max_len: int = 1000) -> str:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    for m in rx.finditer(text):
        start = max(0, text.rfind(".", 0, m.start()) + 1)
        end_dot = text.find(".", m.end())
        end_nl = text.find("\n", m.end())
        ends = [x for x in [end_dot, end_nl] if x != -1]
        end = min(ends) if ends else min(len(text), m.end() + max_len)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        if snippet:
            return snippet[:max_len]
    return ""


def extract_numeric_near(text: str, anchor_pattern: str, value_pattern: str, lo: float, hi: float) -> tuple[float | None, str]:
    anchor = re.compile(anchor_pattern, flags=re.IGNORECASE)
    value = re.compile(value_pattern, flags=re.IGNORECASE)
    for m in anchor.finditer(text):
        window = text[max(0, m.start() - 180): min(len(text), m.end() + 240)]
        vm = value.search(window)
        if not vm:
            continue
        try:
            out = float(vm.group(1).replace(",", ""))
        except Exception:
            continue
        if lo <= out <= hi:
            return out, re.sub(r"\s+", " ", window).strip()[:900]
    return None, ""


def extract_process_method(text: str, context_pattern: str) -> tuple[str | None, str]:
    ctx = re.compile(context_pattern, flags=re.IGNORECASE)
    for m in ctx.finditer(text):
        window = text[max(0, m.start() - 220): min(len(text), m.end() + 280)]
        hits = []
        for name, pat in PROCESS_TERMS:
            if re.search(pat, window, flags=re.IGNORECASE):
                hits.append(name)
        if hits:
            return "; ".join(dict.fromkeys(hits)), re.sub(r"\s+", " ", window).strip()[:900]
    return None, ""


def extract_solvents_and_ratio(text: str) -> tuple[str | None, str | None, str]:
    snippet = first_sentence_with(text, r"\b(?:DMF|DMSO|NMP|GBL|chlorobenzene|toluene|antisolvent|solvent)\b")
    solvents = find_terms(snippet or text[:16000], SOLVENT_TERMS)
    ratio = None
    ratio_match = re.search(
        r"\b(?:DMF|DMSO|NMP|GBL|solvent[s]?)\b[^.\n]{0,100}?\b(\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)?)\b",
        snippet or text[:16000],
        flags=re.IGNORECASE,
    )
    if ratio_match:
        ratio = ratio_match.group(1).replace(" ", "")
    return solvents, ratio, snippet


def extract_cell_area(text: str) -> tuple[float | None, str]:
    patterns = [
        r"\b(?:active|aperture|device|cell|masked)\s+area\b[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*cm\s*(?:2|\^2|-2)?",
        r"\b(\d+(?:\.\d+)?)\s*cm\s*(?:2|\^2)\b[^.\n]{0,80}?\b(?:active|aperture|device|cell|masked)\s+area\b",
    ]
    for pat in patterns:
        val, snippet = extract_numeric_near(text, r"\b(?:active|aperture|device|cell|masked)\s+area\b", pat, 0.001, 20.0)
        if val is not None:
            return val, snippet
    return None, ""


def extract_shap_priority_fields(text: str) -> dict[str, tuple[object, str]]:
    out: dict[str, tuple[object, str]] = {}

    temp, temp_snip = extract_numeric_near(
        text,
        r"\banneal(?:ing|ed)?\b",
        r"(\d{2,3}(?:\.\d+)?)\s*(?:°\s*C|degrees?\s*C|C\b)",
        40,
        250,
    )
    if temp is not None:
        out["Perovskite_deposition_thermal_annealing_temperature"] = (temp, temp_snip)

    anneal_time, time_snip = extract_numeric_near(
        text,
        r"\banneal(?:ing|ed)?\b",
        r"(\d{1,4}(?:\.\d+)?)\s*(?:min|minutes)\b",
        0.1,
        600,
    )
    if anneal_time is not None:
        out["Perovskite_deposition_thermal_annealing_time"] = (anneal_time, time_snip)

    solvents, ratio, solvent_snip = extract_solvents_and_ratio(text)
    if solvents:
        out["Perovskite_deposition_solvents"] = (solvents, solvent_snip)
    if ratio:
        out["Perovskite_deposition_solvents_mixing_ratios"] = (ratio, solvent_snip)

    area, area_snip = extract_cell_area(text)
    if area is not None:
        out["Cell_area_measured"] = (area, area_snip)

    etl_thick, etl_snip = extract_numeric_near(text, r"\b(?:ETL|electron transport layer|SnO2|TiO2|C60|PCBM)\b", r"(\d{1,4}(?:\.\d+)?)\s*nm\b", 0.1, 1000)
    htl_thick, htl_snip = extract_numeric_near(text, r"\b(?:HTL|hole transport layer|Spiro|PTAA|PEDOT:PSS|NiOx)\b", r"(\d{1,4}(?:\.\d+)?)\s*nm\b", 0.1, 1000)
    back_thick, back_snip = extract_numeric_near(text, r"\b(?:Au|Ag|Al|Cu|back contact|electrode)\b", r"(\d{1,4}(?:\.\d+)?)\s*nm\b", 0.1, 1000)
    if etl_thick is not None:
        out["ETL_thickness"] = (etl_thick, etl_snip)
    if htl_thick is not None:
        out["HTL_thickness_list"] = (htl_thick, htl_snip)
    if back_thick is not None:
        out["Backcontact_thickness_list"] = (back_thick, back_snip)

    etl_proc, etl_proc_snip = extract_process_method(text, r"\b(?:ETL|electron transport layer|SnO2|TiO2|C60|PCBM)\b")
    htl_proc, htl_proc_snip = extract_process_method(text, r"\b(?:HTL|hole transport layer|Spiro|PTAA|PEDOT:PSS|NiOx)\b")
    back_proc, back_proc_snip = extract_process_method(text, r"\b(?:Au|Ag|Al|Cu|back contact|electrode)\b")
    if etl_proc:
        out["ETL_deposition_procedure"] = (etl_proc, etl_proc_snip)
    if htl_proc:
        out["HTL_deposition_procedure"] = (htl_proc, htl_proc_snip)
    if back_proc:
        out["Backcontact_deposition_procedure"] = (back_proc, back_proc_snip)

    additives = find_terms(text[:24000], ADDITIVE_TERMS)
    if additives:
        out["Perovskite_additives_compounds"] = (additives, first_sentence_with(text, r"\b(?:additive|MACl|FACl|PEAI|BAI|passivat|dop)\b"))

    return out


def load_slug_text(work_dir: Path, slug: str, max_chars_per_file: int) -> str:
    paths = []
    for sub in ["text", "paper_summaries_text"]:
        folder = work_dir / sub
        if folder.exists():
            paths.extend(sorted(folder.glob(f"{slug}*.txt")))
    paths = sorted(
        paths,
        key=lambda p: (
            "performance" not in p.name.lower(),
            "reference_device" not in p.name.lower(),
            "summary" not in p.name.lower(),
            "stability" not in p.name.lower(),
            p.name,
        ),
    )
    parts = []
    for path in paths[:24]:
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file])
        except Exception:
            pass
    return "\n".join(parts)


def fill_row(row: pd.Series, text: str, slug: str, min_pce_score: int) -> tuple[pd.Series, list[str], dict]:
    changes = []
    perf = extract_best_performance(text, min_score=min_pce_score)
    stability = extract_best_stability(text)
    has_or_recovered_pce = model_range_pce(row.get("JV_default_PCE"))
    if perf and not model_range_pce(row.get("JV_default_PCE")):
        row["JV_default_PCE"] = perf["value"]
        row["JV_default_PCE_scan_direction"] = row.get("JV_default_PCE_scan_direction") if present(row.get("JV_default_PCE_scan_direction")) else "unspecified_text_mined"
        row["_lit_agent_target_recovery"] = 1
        row["_lit_agent_target_recovery_method"] = "source_text_regex_high_confidence"
        row["_lit_agent_target_recovery_score"] = perf["score"]
        row["_lit_agent_target_recovery_evidence"] = perf["snippet"][:900]
        changes.append(f"JV_default_PCE={perf['value']}")
        has_or_recovered_pce = True
        if perf.get("voc") is not None and not present(row.get("JV_default_Voc")):
            row["JV_default_Voc"] = perf["voc"]
            changes.append(f"JV_default_Voc={perf['voc']}")
        if perf.get("jsc") is not None and not present(row.get("JV_default_Jsc")):
            row["JV_default_Jsc"] = perf["jsc"]
            changes.append(f"JV_default_Jsc={perf['jsc']}")
        if perf.get("ff") is not None and not present(row.get("JV_default_FF")):
            ff = perf["ff"] / 100.0 if perf["ff"] and perf["ff"] > 1.0 else perf["ff"]
            row["JV_default_FF"] = ff
            changes.append(f"JV_default_FF={ff}")

    feature_text = f"{perf.get('snippet', '') if perf else ''}\n{text[:12000]}"
    if has_or_recovered_pce:
        formula = extract_formula(text)
        if formula and not present(row.get("Perovskite_composition_short_form")):
            row["Perovskite_composition_short_form"] = formula
            row["Perovskite_composition_long_form"] = formula
            changes.append(f"Perovskite_composition_short_form={formula}")

        etl = find_terms(feature_text, ["TiO2", "SnO2", "PCBM", "C60", "ZnO", "NiOx", "BCP"])
        htl = find_terms(feature_text, ["Spiro-OMeTAD", "PTAA", "PEDOT:PSS", "NiOx", "CuSCN", "P3HT", "MeO-2PACz", "2PACz"])
        back = find_terms(feature_text, ["Au", "Ag", "Al", "Cu", "ITO", "FTO", "carbon"])
        if etl and not present(row.get("ETL_stack_sequence")):
            row["ETL_stack_sequence"] = etl
            changes.append(f"ETL_stack_sequence={etl}")
        if htl and not present(row.get("HTL_stack_sequence")):
            row["HTL_stack_sequence"] = htl
            changes.append(f"HTL_stack_sequence={htl}")
        if back and not present(row.get("Backcontact_stack_sequence")):
            row["Backcontact_stack_sequence"] = back
            changes.append(f"Backcontact_stack_sequence={back}")
        if not present(row.get("Cell_architecture")):
            low = feature_text.lower()
            if re.search(r"\b(inverted|p-i-n|pin)\b", low):
                row["Cell_architecture"] = "inverted"
                changes.append("Cell_architecture=inverted")
            elif re.search(r"\b(regular|n-i-p|nip|mesoscopic|planar)\b", low):
                row["Cell_architecture"] = "regular"
                changes.append("Cell_architecture=regular")
        if etl and htl and back and not present(row.get("Cell_stack_sequence")):
            row["Cell_stack_sequence"] = f"{etl} / perovskite / {htl} / {back}"
            changes.append(f"Cell_stack_sequence={row['Cell_stack_sequence']}")

        shap_priority_changes = []
        shap_priority_evidence = []
        shap_priority = extract_shap_priority_fields(feature_text)
        for field in SHAP_PRIORITY_FIELDS:
            if field not in shap_priority:
                continue
            value, evidence = shap_priority[field]
            if present(value) and not present(row.get(field)):
                row[field] = value
                shap_priority_changes.append(f"{field}={value}")
                if evidence:
                    shap_priority_evidence.append(f"{field}: {evidence}")
        if shap_priority_changes:
            row["_lit_agent_shap_priority_recovery"] = 1
            row["_lit_agent_shap_priority_recovery_fields"] = "; ".join(shap_priority_changes)
            row["_lit_agent_shap_priority_recovery_evidence"] = "\n---\n".join(shap_priority_evidence)[:1800]
            changes.extend(shap_priority_changes)

    stability_changes = []
    has_stability_target_candidate = bool(
        stability.get("t80")
        or stability.get("t95")
        or stability.get("retention")
        or stability.get("after_1000h")
    )
    if stability.get("t80") and not present(row.get("Stability_PCE_T80")):
        row["Stability_PCE_T80"] = stability["t80"]["value"]
        stability_changes.append(f"Stability_PCE_T80={stability['t80']['value']}")
    if stability.get("t95") and not present(row.get("Stability_PCE_T95")):
        row["Stability_PCE_T95"] = stability["t95"]["value"]
        stability_changes.append(f"Stability_PCE_T95={stability['t95']['value']}")

    retention = stability.get("retention") or {}
    if retention:
        if not present(row.get("Stability_PCE_end_of_experiment")):
            row["Stability_PCE_end_of_experiment"] = retention["retention_pct"]
            stability_changes.append(f"Stability_PCE_end_of_experiment={retention['retention_pct']}")
        if not present(row.get("Stability_time_total_exposure")):
            row["Stability_time_total_exposure"] = retention["hours"]
            stability_changes.append(f"Stability_time_total_exposure={retention['hours']}")
        if (
            950 <= float(retention["hours"]) <= 1050
            and not present(row.get("Stability_PCE_after_1000_h"))
        ):
            row["Stability_PCE_after_1000_h"] = retention["retention_pct"]
            stability_changes.append(f"Stability_PCE_after_1000_h={retention['retention_pct']}")

    conditions = stability.get("conditions") or {}
    if not has_stability_target_candidate:
        conditions = {}
    if conditions:
        if conditions.get("temperature_c") is not None and not present(row.get("Stability_temperature_range")):
            row["Stability_temperature_range"] = conditions["temperature_c"]
            stability_changes.append(f"Stability_temperature_range={conditions['temperature_c']}")
        if conditions.get("rh_percent") is not None and not present(row.get("Stability_relative_humidity_average_value")):
            row["Stability_relative_humidity_average_value"] = conditions["rh_percent"]
            stability_changes.append(f"Stability_relative_humidity_average_value={conditions['rh_percent']}")
        if conditions.get("light") is not None and not present(row.get("Stability_light_intensity")):
            row["Stability_light_intensity"] = conditions["light"]
            stability_changes.append(f"Stability_light_intensity={conditions['light']}")
        if conditions.get("atmosphere") and not present(row.get("Stability_atmosphere")):
            row["Stability_atmosphere"] = conditions["atmosphere"]
            stability_changes.append(f"Stability_atmosphere={conditions['atmosphere']}")
        if conditions.get("encapsulation") and not present(row.get("Encapsulation")):
            row["Encapsulation"] = conditions["encapsulation"]
            stability_changes.append(f"Encapsulation={conditions['encapsulation']}")

    if stability_changes:
        row["_lit_agent_stability_recovery"] = 1
        evidence_parts = []
        for key in ["t80", "t95", "retention"]:
            if stability.get(key):
                evidence_parts.append(stability[key].get("snippet", ""))
        row["_lit_agent_stability_recovery_evidence"] = "\n---\n".join(evidence_parts)[:1400]
        row["_lit_agent_stability_recovery_changes"] = "; ".join(stability_changes)
        changes.extend(stability_changes)

    if changes:
        note = "source-text finalization for explicit targets and SHAP-priority model fields; verify before final quantitative use"
        existing = str(row.get("_lit_agent_recovery_notes") or "").strip()
        row["_lit_agent_recovery_notes"] = (existing + "; " + note).strip("; ")
        row["_lit_agent_target_recovery_changes"] = "; ".join(changes)
    return row, changes, perf


def fieldnames_for(paths: list[Path], rows: list[dict]) -> list[str]:
    names = []
    seen = set()
    for path in paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for name in reader.fieldnames or []:
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
        except Exception:
            pass
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--apply", action="store_true", help="Modify per-paper CSVs and rebuild csv/all_records.csv. A timestamped backup is created.")
    ap.add_argument("--max-chars-per-file", type=int, default=50000)
    ap.add_argument("--min-pce-score", type=int, default=11,
                    help="Minimum source-text confidence score required before filling JV_default_PCE. Default 11 keeps only strong device-local snippets.")
    ap.add_argument("--validated-report", type=Path,
                    help="Optional validated target_recovery_report CSV. If provided, only validation_status=pass rows are changed.")
    args = ap.parse_args()

    csv_dir = args.work_dir / "csv"
    report_dir = args.work_dir / "target_recovery_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.work_dir / f"csv_backup_before_target_recovery_{stamp}"

    per_paper_csvs = sorted(p for p in csv_dir.glob("*.csv") if p.name.lower() not in AGGREGATE_NAMES)
    all_rows = []
    report_rows = []
    changed_files = 0
    changed_rows = 0
    changed_pce_rows = 0
    changed_stability_rows = 0
    changed_shap_priority_rows = 0
    rows_after_model_range_pce = 0
    rows_after_stability_target = 0
    validation_pass_keys = load_validation_pass_keys(args.validated_report)
    validation_blocked_rows = 0

    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in csv_dir.glob("*.csv"):
            shutil.copy2(p, backup_dir / p.name)

    for csv_path in per_paper_csvs:
        slug = csv_path.stem
        df = read_csv(csv_path)
        if df.empty:
            continue
        text = load_slug_text(args.work_dir, slug, args.max_chars_per_file)
        file_changed = False
        new_rows = []
        for idx, row in df.iterrows():
            original_row = row.copy()
            before_pce = row.get("JV_default_PCE")
            new_row, changes, perf = fill_row(row.copy(), text, slug, args.min_pce_score)
            if changes and validation_pass_keys is not None and (slug, int(idx)) not in validation_pass_keys:
                new_row = original_row.copy()
                changes = []
                validation_blocked_rows += 1
            after_has_model_pce = model_range_pce(new_row.get("JV_default_PCE"))
            if after_has_model_pce:
                rows_after_model_range_pce += 1
            pce_changed = (
                not model_range_pce(before_pce)
                and after_has_model_pce
            )
            stability_changed = any(str(change).startswith("Stability_") for change in changes)
            has_stability_after = any(
                present(new_row.get(col))
                for col in [
                    "Stability_PCE_T80",
                    "Stability_PCE_T95",
                    "Stability_PCE_end_of_experiment",
                    "Stability_PCE_after_1000_h",
                ]
            )
            if has_stability_after:
                rows_after_stability_target += 1
            new_rows.append(new_row.to_dict())
            if changes:
                file_changed = True
                changed_rows += 1
            if pce_changed:
                changed_pce_rows += 1
            if stability_changed:
                changed_stability_rows += 1
            if present(new_row.get("_lit_agent_shap_priority_recovery")):
                changed_shap_priority_rows += 1
            report_rows.append({
                "paper_slug": slug,
                "row_index": int(idx),
                "before_JV_default_PCE": before_pce,
                "after_JV_default_PCE": new_row.get("JV_default_PCE"),
                "after_Stability_PCE_T80": new_row.get("Stability_PCE_T80"),
                "after_Stability_PCE_T95": new_row.get("Stability_PCE_T95"),
                "after_Stability_PCE_end_of_experiment": new_row.get("Stability_PCE_end_of_experiment"),
                "after_Stability_PCE_after_1000_h": new_row.get("Stability_PCE_after_1000_h"),
                "after_Stability_time_total_exposure": new_row.get("Stability_time_total_exposure"),
                "changed": bool(changes),
                "pce_changed_to_model_range": bool(pce_changed),
                "stability_changed": bool(stability_changed),
                "has_model_range_pce_after": bool(after_has_model_pce),
                "has_stability_target_after": bool(has_stability_after),
                "changes": "; ".join(changes),
                "candidate_score": perf.get("score") if perf else None,
                "candidate_kind": perf.get("kind") if perf else None,
                "candidate_evidence": perf.get("snippet") if perf else None,
                "stability_recovery_evidence": new_row.get("_lit_agent_stability_recovery_evidence"),
                "shap_priority_recovery": new_row.get("_lit_agent_shap_priority_recovery"),
                "shap_priority_recovery_fields": new_row.get("_lit_agent_shap_priority_recovery_fields"),
                "shap_priority_recovery_evidence": new_row.get("_lit_agent_shap_priority_recovery_evidence"),
            })
        out_df = pd.DataFrame(new_rows)
        all_rows.extend(new_rows)
        if args.apply and file_changed:
            write_csv(out_df, csv_path)
            changed_files += 1

    report = pd.DataFrame(report_rows)
    report_path = report_dir / "target_recovery_report.csv"
    write_csv(report, report_path)

    if args.apply and all_rows:
        fieldnames = fieldnames_for(per_paper_csvs, all_rows)
        target = csv_dir / "all_records.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        manager = csv_dir / "all_records_from_manager.csv"
        shutil.copy2(target, manager)

    summary = pd.DataFrame([{
        "paper_csvs": len(per_paper_csvs),
        "rows_seen": len(all_rows),
        "changed_rows": changed_rows,
        "changed_pce_rows": changed_pce_rows,
        "changed_stability_rows": changed_stability_rows,
        "changed_shap_priority_rows": changed_shap_priority_rows,
        "rows_after_model_range_pce": rows_after_model_range_pce,
        "rows_after_stability_target": rows_after_stability_target,
        "changed_files": changed_files,
        "applied": bool(args.apply),
        "min_pce_score": int(args.min_pce_score),
        "backup_dir": str(backup_dir) if args.apply else "",
        "report_csv": str(report_path),
        "validated_report": str(args.validated_report) if args.validated_report else "",
        "validation_blocked_rows": validation_blocked_rows,
    }])
    write_csv(summary, report_dir / "target_recovery_summary.csv")
    print(summary.to_string(index=False))
    if not args.apply:
        print("Dry run only. Re-run with --apply to modify CSVs.")


if __name__ == "__main__":
    main()
