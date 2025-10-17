#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1日CSV（docs/outputs/<index>_1d.csv）から当日最初と最新の有効値を取り、
A%（騰落率）を計算して以下を更新します。

- docs/outputs/<index>_post_intraday.txt  … X用の1行テキスト
  例: "AIN10 1d: Δ=N/A (level) A%=+1.23% (basis=open valid=2025-10-16 first->latest)"

- docs/outputs/<index>_stats.json          … サイト用サマリー
  例: {"index_key":"ain10","pct_1d":1.23,"delta_level":null,"scale":"level","basis":"open","updated_at":"...Z"}

※Δ（レベル差）は常に N/A 固定（ご要望どおり）。
"""

import os
import json
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

def pick_time_col(df: pd.DataFrame):
    for c in ["Datetime","datetime","timestamp","time","date","ts"]:
        if c in df.columns:
            return c
    return df.columns[0]

def pick_value_col(df: pd.DataFrame, index_key: str):
    cands = [
        index_key, index_key.upper(), index_key.title(),
        index_key.replace("_","-").upper(),
        "AIN-10", "AIN10",
        "value","Value","index","score","close","price","y"
    ]
    for c in cands:
        if c in df.columns:
            return c
    return df.columns[-1]

def first_last_valid(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return None, None
    first = s[s.notna()].iloc[0]
    last  = s[s.notna()].iloc[-1]
    return float(first), float(last)

def sign_fmt(x: float, digits=2) -> str:
    return f"{x:+.{digits}f}"

def compute_pct_1d(csv_path: str, index_key: str):
    if not os.path.exists(csv_path):
        return None, None, None  # pct, first_ts, last_ts

    df = pd.read_csv(csv_path)
    tcol = pick_time_col(df)
    vcol = pick_value_col(df, index_key)
    if tcol not in df.columns or vcol not in df.columns:
        return None, None, None

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")

    if df[vcol].dropna().empty:
        return None, None, None

    first_val = df[vcol].dropna().iloc[0]
    last_val  = df[vcol].dropna().iloc[-1]
    first_ts  = df[tcol].iloc[ df[vcol].first_valid_index() ]
    last_ts   = df[tcol].iloc[ df[vcol].last_valid_index() ]

    # 騰落率（%）: (last-first) / abs(first) * 100
    if first_val == 0 or np.isfinite(first_val) is False or np.isfinite(last_val) is False:
        return None, first_ts, last_ts
    pct = (last_val - first_val) / abs(first_val) * 100.0
    return float(pct), first_ts, last_ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/<index>_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", default="open", choices=["open","prev_close","first_valid"],
                    help="表記用。実計算は first_valid→latest を用います（ご要望に合わせて open 同等）。")
    args = ap.parse_args()

    index_key = args.index_key.lower()
    pct, first_ts, last_ts = compute_pct_1d(args.csv, index_key)

    # 出力の整形
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    title_key = "AIN10" if index_key in ("ain10","ain_10","ain-10") else index_key.upper()

    # テキスト（Δは常に N/A 固定）
    if pct is None or first_ts is None or last_ts is None:
        text = f"{title_key} 1d: Δ=N/A (level) A%=N/A (basis=n/a valid=n/a)"
        pct_json = None
    else:
        day_str = first_ts.date().isoformat()
        text = f"{title_key} 1d: Δ=N/A (level) A%={sign_fmt(pct,2)}% (basis={args.basis} valid={day_str} first->latest)"
        pct_json = round(pct, 6)

    # JSON
    stats = {
        "index_key": index_key,
        "pct_1d": pct_json,
        "delta_level": None,      # ← 常に N/A（null）
        "scale": "level",
        "basis": args.basis if pct_json is not None else "n/a",
        "updated_at": updated_at,
    }

    os.makedirs(os.path.dirname(args.out_text), exist_ok=True)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
