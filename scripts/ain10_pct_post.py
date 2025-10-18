#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd

MIN_ABS_BASE = 0.10
SEARCH_FROM = "09:35"

def iso_now() -> str:
    ts = pd.Timestamp.utcnow()
    ts = ts.tz_localize("UTC")
    return ts.isoformat().replace("+00:00", "Z")

def read_1d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def pick_base(df: pd.DataFrame) -> tuple[float | None, str]:
    """09:35以降で最初の|val|>=MIN_ABS_BASEを基準にする。なければ先頭値が十分大きければ使う。"""
    if df.empty:
        return None, "n/a"
    day = df["ts"].dt.floor("D").max()
    start = pd.Timestamp(f"{day.date()} {SEARCH_FROM}")
    cand = df[df["ts"] >= start]
    cand = cand[cand["val"].abs() >= MIN_ABS_BASE]
    if not cand.empty:
        return float(cand.iloc[0]["val"]), f"first_nonzero@{SEARCH_FROM}"
    first_val = float(df.iloc[0]["val"])
    if abs(first_val) >= MIN_ABS_BASE:
        return first_val, "open"
    return None, "no_pct_col"

def percent_change(base: float, last: float) -> float | None:
    try:
        if base is None or abs(base) < MIN_ABS_BASE:
            return None
        return (last - base) / abs(base) * 100.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", required=False, default=None, help="optional basis label (ignored)")  # 👈 ← 修正点
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val = None
    delta_level = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])
        delta_level = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        base, basis_note = pick_base(df)
        pct_val = percent_change(base, last_val)

    # basis引数が渡されていた場合は優先的にメモに反映（値は無視）
    if args.basis:
        basis_note = args.basis

    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
