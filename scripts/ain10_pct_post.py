#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIN10 1日騰落率算出スクリプト（最終安定版）

- docs/outputs/<index>_1d.csv を入力
- first→latest の差分で A% を計算
- Δ=N/A固定, JSON/テキストを出力
"""

import os
import json
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

def detect_time_col(df: pd.DataFrame):
    for c in df.columns:
        if any(k in c.lower() for k in ["datetime", "timestamp", "time", "date", "ts"]):
            return c
    return df.columns[0]

def detect_value_col(df: pd.DataFrame, index_key: str):
    # 既知候補（大文字・小文字・ハイフン対応）
    key_variants = [
        index_key, index_key.lower(), index_key.upper(),
        index_key.replace("_", "-").upper(),
        index_key.replace("_", "-").lower(),
        "AIN10", "AIN-10",
        "value", "index", "score", "close", "price", "y"
    ]
    for c in df.columns:
        if c in key_variants or c.lower() in [v.lower() for v in key_variants]:
            return c
    # 最後の列を fallback に
    return df.columns[-1]

def compute_pct(csv_path: str, index_key: str):
    if not os.path.exists(csv_path):
        return None, None, None

    df = pd.read_csv(csv_path)
    if df.empty or len(df.columns) < 2:
        return None, None, None

    tcol = detect_time_col(df)
    vcol = detect_value_col(df, index_key)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")

    df = df.dropna(subset=[vcol])
    if df.empty:
        return None, None, None

    first_val, last_val = df[vcol].iloc[0], df[vcol].iloc[-1]
    first_ts, last_ts = df[tcol].iloc[0], df[tcol].iloc[-1]

    if not np.isfinite(first_val) or first_val == 0:
        return None, first_ts, last_ts

    pct = (last_val - first_val) / abs(first_val) * 100
    return float(pct), first_ts, last_ts

def fmt_sign(x, d=2):
    return f"{x:+.{d}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", default="open")
    args = ap.parse_args()

    index_key = args.index_key.lower()
    pct, first_ts, last_ts = compute_pct(args.csv, index_key)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = "AIN10" if "ain" in index_key else index_key.upper()

    if pct is None:
        text = f"{name} 1d: Δ=N/A (level) A%=N/A (basis=n/a valid=n/a)"
        stats = {
            "index_key": index_key,
            "pct_1d": None,
            "delta_level": None,
            "scale": "level",
            "basis": "n/a",
            "updated_at": now_utc,
        }
    else:
        day_str = first_ts.date().isoformat()
        text = f"{name} 1d: Δ=N/A (level) A%={fmt_sign(pct)}% (basis={args.basis} valid={day_str} first->latest)"
        stats = {
            "index_key": index_key,
            "pct_1d": round(pct, 6),
            "delta_level": None,
            "scale": "level",
            "basis": args.basis,
            "updated_at": now_utc,
        }

    os.makedirs(os.path.dirname(args.out_text), exist_ok=True)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
