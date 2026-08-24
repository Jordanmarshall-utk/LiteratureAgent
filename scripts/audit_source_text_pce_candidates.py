from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PCE_COLS = [
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
]


PCE_PATTERNS = [
    r"\bPCE\s*(?:of|=|:|reached|reaches|achieved|achieves|up to|exceeding|over|above|as high as|was|is)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\bpower conversion efficienc(?:y|ies)\s*(?:of|=|:|reached|reaches|achieved|achieves|up to|exceeding|over|above|as high as|was|is)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\b(?:champion|best|maximum|certified|record|device|cell|PSC|solar cell)[^.\n]{0,120}\b(?:efficienc(?:y|ies)|PCE)[^.\n]{0,80}?(\d{1,2}(?:\.\d+)?)\s*%",
    r"\b(?:champion|best|maximum|certified|record|device|cell|PSC|solar cell)[^.\n]{0,180}?(\d{1,2}(?:\.\d+)?)\s*%\s*(?:PCE|power conversion efficiency|efficiency)\b",
    r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:PCE|power conversion efficiency|photoconversion efficiency)\b",
]


BAD_CONTEXT = re.compile(
    r"\b(retention|retained|humidity|RH|relative humidity|degradation|yield|quantum yield|PLQY|EQE|IPCE|transmittance|absorbance)\b",
    flags=re.IGNORECASE,
)


GOOD_CONTEXT = re.compile(
    r"\b(PCE|power conversion|photoconversion|device|cell|solar|photovoltaic|PSC|champion|certified|Voc|Jsc|fill factor|FF)\b",
    flags=re.IGNORECASE,
)


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def pce_in_model_range(row: pd.Series) -> bool:
    for col in PCE_COLS:
        if col not in row.index:
            continue
        val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(val) and 5 <= float(val) <= 30:
            return True
    return False


def extract_candidate_pces(text: str) -> list[dict]:
    text = re.sub(r"\s+", " ", str(text or ""))
    hits = []
    for pattern in PCE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = float(match.group(1))
            if not (5 <= value <= 30):
                continue
            start = max(0, match.start() - 220)
            end = min(len(text), match.end() + 220)
            window = text[start:end]
            if BAD_CONTEXT.search(window) and not re.search(r"\b(PCE|power conversion|device|cell|PSC|solar)\b", match.group(0), flags=re.IGNORECASE):
                continue
            if not GOOD_CONTEXT.search(window):
                continue
            hits.append({
                "candidate_pce": value,
                "snippet": window.strip(),
            })
    dedup = {}
    for hit in sorted(hits, key=lambda d: d["candidate_pce"], reverse=True):
        dedup.setdefault(hit["candidate_pce"], hit)
    return list(dedup.values())


def load_slug_text(work_dir: Path, slug: str, max_chars_per_file: int) -> str:
    paths = []
    for sub in ["text", "paper_summaries_text"]:
        folder = work_dir / sub
        if folder.exists():
            paths.extend(sorted(folder.glob(f"{slug}*.txt")))
    # Performance/reference chunks carry the highest signal; keep them first.
    paths = sorted(
        paths,
        key=lambda p: (
            "performance" not in p.name.lower(),
            "reference_device" not in p.name.lower(),
            "summary" not in p.name.lower(),
            p.name,
        ),
    )
    parts = []
    for path in paths[:20]:
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_file])
        except Exception:
            pass
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-chars-per-file", type=int, default=50000)
    args = ap.parse_args()

    csv_dir = args.work_dir / "csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        if csv_path.name.lower() in {"all_records.csv", "all_records_from_manager.csv", "paper_summaries_from_manager.csv"}:
            continue
        slug = csv_path.stem
        df = read_csv(csv_path)
        raw_has_pce = False
        raw_rows = len(df)
        if not df.empty:
            raw_has_pce = bool(df.apply(pce_in_model_range, axis=1).any())
        text = load_slug_text(args.work_dir, slug, args.max_chars_per_file)
        candidates = extract_candidate_pces(text)
        best = candidates[0] if candidates else {}
        rows.append({
            "paper_slug": slug,
            "raw_rows": raw_rows,
            "raw_has_model_range_pce_any_col": raw_has_pce,
            "source_text_chars": len(text),
            "candidate_pce_count": len(candidates),
            "best_candidate_pce": best.get("candidate_pce"),
            "best_candidate_snippet": best.get("snippet"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "source_text_pce_candidate_audit.csv", index=False)
    summary = pd.DataFrame([{
        "paper_csvs": len(out),
        "raw_paper_csvs_with_model_range_pce": int(out["raw_has_model_range_pce_any_col"].sum()) if not out.empty else 0,
        "paper_csvs_with_source_text_candidate_pce": int((out["candidate_pce_count"] > 0).sum()) if not out.empty else 0,
        "paper_csvs_with_candidate_pce_but_raw_missing": int(((out["candidate_pce_count"] > 0) & ~out["raw_has_model_range_pce_any_col"]).sum()) if not out.empty else 0,
    }])
    summary.to_csv(args.out_dir / "source_text_pce_candidate_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_dir / 'source_text_pce_candidate_audit.csv'}")


if __name__ == "__main__":
    main()
