#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd
from typing import Optional

EPS = 1e-6
MINUTES_AFTER_OPEN = 30  # 最初の30分はノイズ扱い

def iso_now() -> str:
    """UTCのISO8601(Z付き)"""
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")

def read_1d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def find_stable_open(df: pd.DataFrame) -> tuple[Optional[float], str]:
    """
    開始直後ではなく、安定した最初の値を基準にする
    """
    if df.empty:
        return None, "n/a"

    t0 = df.iloc[0]["ts"]
    later = t0 + pd.Timedelta(minutes=MINUTES_AFTER_OPEN)

    # 開始30分以降の最初の値を使う（なければ中央値）
    m = df["ts"] >= later
    if m.any():
        val = float(df[m].iloc[0]["val"])
        ts = pd.to_datetime(df[m].iloc[0]["ts"])
        return val, f"stable@{ts.strftime('%H:%M')}"
    else:
        # 開始から終了までの中央値（全体の中間値）を採用
        median_row = df.iloc[len(df)//2]
        return float(median_row["val"]), f"median@{pd.to_datetime(median_row['ts']).strftime('%H:%M')}"

def percent_change(first: float, last: float) -> Optional[float]:
    try:
        if first is None or last is None:
            return None
        if pd.isna(first) or pd.isna(last):
            return None
        if abs(float(first)) < EPS:
            return None
        return (float(last) - float(first)) / abs(float(first)) * 100.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", choices=["open", "prev_close"], default="open")
    ap.add_argument("--history", required=False)
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: Optional[float] = None
    delta_level: Optional[float] = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        last_val  = float(last_row["val"])
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        base_val, base_note = find_stable_open(df)
        basis_note = base_note

        delta_level = last_val - float(first_row["val"])
        pct_val = percent_change(base_val, last_val)

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
