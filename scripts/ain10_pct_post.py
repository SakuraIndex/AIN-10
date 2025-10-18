#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

MIN_ABS_BASE = 0.10          # 基準の最小絶対値
SEARCH_FROM = "09:35"        # この時刻以降で最初の非ゼロを基準にする

def iso_now() -> str:
    ts = pd.Timestamp.utcnow()
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
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
    """09:35以降で最初の|val|>=MIN_ABS_BASE を基準に選ぶ。なければ最初の値で判定。"""
    if df.empty:
        return None, "n/a"
    day = df["ts"].dt.floor("D").max()
    start = pd.Timestamp(f"{day.date()} {SEARCH_FROM}")
    cand = df[df["ts"] >= start]
    cand = cand[ cand["val"].abs() >= MIN_ABS_BASE ]
    if not cand.empty:
        v = float(cand.iloc[0]["val"])
        return v, f"first_nonzero@{SEARCH_FROM}"
    # フォールバック：当日の最初
    v0 = float(df.iloc[0]["val"])
    if abs(v0) >= MIN_ABS_BASE:
        return v0, "open"
    return None, "no_pct_col"

def percent_change(base: float, last: float) -> float | None:
    try:
        if base is None:
            return None
        if abs(base) < MIN_ABS_BASE:
            return None
        return (float(last) - float(base)) / abs(float(base)) * 100.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
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
