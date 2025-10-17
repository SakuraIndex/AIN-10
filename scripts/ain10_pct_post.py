#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import math
import pandas as pd

def iso_now() -> str:
    """常にUTCのISO8601（Z付き）を返す。"""
    ts = pd.Timestamp.utcnow()
    # tz-aware なら convert、naive なら localize
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

def percent_change(first: float, last: float) -> float | None:
    """NaNや0割防止付きの単純騰落率(%単位)"""
    try:
        if first is None or last is None:
            return None
        if pd.isna(first) or pd.isna(last):
            return None
        if abs(float(first)) < 1e-9:  # ← 許容範囲を緩く
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
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val = None
    delta_level = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        # 1dの最初と最後
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])
        delta_level = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        if args.basis == "open":
            basis_note = "open"
            pct_val = percent_change(first_val, last_val)
        else:
            # 今回用に open と同一（必要になったらここで差し替え）
            basis_note = "prev_close"
            pct_val = percent_change(first_val, last_val)

    # --- TXT 出力
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力
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
