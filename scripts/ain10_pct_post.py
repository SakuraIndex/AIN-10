#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
import argparse
import pandas as pd
from datetime import timezone

def load_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")
    ts_col, val_col = df.columns[0], df.columns[1]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def last_trading_day(df: pd.DataFrame):
    return df["ts"].dt.floor("D").max()

def compute_basis_value(df: pd.DataFrame, basis: str):
    """
    basis:
      - 'open'       : 当日(最新日)の最初の値
      - 'prev_close' : 前営業日の最後の値
      - 'prev_any'   : 当日最初の点の直前の値（直近のティック）
    """
    if df.empty:
        return None, None, None

    d_last = last_trading_day(df)
    day_df = df[df["ts"].dt.floor("D") == d_last].sort_values("ts")
    if day_df.empty:
        return None, None, None

    first = day_df.iloc[0]
    last = day_df.iloc[-1]

    if basis == "open":
        base = first["val"]
        base_ts = first["ts"]
    elif basis == "prev_close":
        prev_df = df[df["ts"] < first["ts"]]
        if prev_df.empty:
            return None, first["ts"], last["ts"]
        base = prev_df.iloc[-1]["val"]
        base_ts = prev_df.iloc[-1]["ts"]
    elif basis == "prev_any":
        prev_df = df[df["ts"] < first["ts"]]
        if prev_df.empty:
            return None, first["ts"], last["ts"]
        base = prev_df.iloc[-1]["val"]
        base_ts = prev_df.iloc[-1]["ts"]
    else:
        raise ValueError("invalid basis")

    return (float(base) if base is not None else None), pd.Timestamp(base_ts), pd.Timestamp(last["ts"])

def compute_pct_1d(df: pd.DataFrame, basis: str):
    base, ts0, ts1 = compute_basis_value(df, basis)
    if base is None or base == 0:
        return None, None, ts0, ts1

    d_last = last_trading_day(df)
    day_df = df[df["ts"].dt.floor("D") == d_last].sort_values("ts")
    if day_df.empty:
        return None, None, ts0, ts1
    last_val = float(day_df.iloc[-1]["val"])
    delta_level = last_val - base
    pct = (delta_level / abs(base)) * 100.0
    return pct, delta_level, ts0, ts1

def iso_now():
    return pd.Timestamp.utcnow().tz_localize("UTC").isoformat()

def main():
    ap = argparse.ArgumentParser(description="Compute 1d percent and write posts/json")
    ap.add_argument("--index-key", required=True, dest="index_key")
    ap.add_argument("--csv", required=True, dest="csv")
    ap.add_argument("--out-json", required=True, dest="out_json")
    ap.add_argument("--out-text", required=True, dest="out_text")
    ap.add_argument("--basis", choices=["open", "prev_close", "prev_any"], default="open")
    args = ap.parse_args()

    df = load_series(Path(args.csv))
    pct, delta_level, ts0, ts1 = compute_pct_1d(df, args.basis)

    # ----- JSON (stats) -----
    stats = {
        "index_key": args.index_key,
        "pct_1d": (None if pct is None else float(pct)),
        "delta_level": (None if delta_level is None else float(delta_level)),
        "scale": "level",
        "basis": args.basis,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(stats, ensure_ascii=False))

    # ----- TEXT (post) -----
    if ts0 is not None and ts1 is not None:
        valid_str = f"{ts0.strftime('%Y-%m-%d %H:%M')}→{ts1.strftime('%Y-%m-%d %H:%M')}"
    else:
        valid_str = "n/a"

    if pct is None or delta_level is None:
        line = f"{args.index_key.upper()} 1d: Δ=N/A (level) A%=N/A (basis={args.basis} valid={valid_str})"
    else:
        line = (
            f"{args.index_key.upper()} 1d: Δ={delta_level:.6f} (level) "
            f"A%={pct:+.2f}% (basis={args.basis} valid={valid_str})"
        )

    Path(args.out_text).write_text(line + "\n")

if __name__ == "__main__":
    main()
