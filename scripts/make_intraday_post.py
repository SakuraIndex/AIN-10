#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

INDEX_KEY_DEFAULT = "ain10"
OUT_DIR = Path("docs/outputs")

# 騰落率ロジック（long_charts/ain10_pct_post と揃える）
EPS = 5.0
CLAMP_PCT = 100.0

def read_intraday(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def choose_baseline(df_day: pd.DataFrame) -> tuple[float | None, str]:
    """open→10:00以降安定→|val|>=EPS→最初の値 の順で基準を決める"""
    if df_day.empty:
        return None, "no_pct_col"
    open_val = float(df_day.iloc[0]["val"])
    if abs(open_val) >= EPS:
        return open_val, "open"
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"]), "stable@10:00"
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"]), "first|val|>=EPS"
    return float(df_day.iloc[0]["val"]), "first_any"

def percent_change(first: float, last: float) -> float | None:
    try:
        if first is None or last is None:
            return None
        denom = max(abs(float(first)), abs(float(last)), EPS)
        pct = (float(last) - float(first)) / denom * 100.0
        if pct > CLAMP_PCT:
            pct = CLAMP_PCT
        elif pct < -CLAMP_PCT:
            pct = -CLAMP_PCT
        return pct
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", default=INDEX_KEY_DEFAULT)
    ap.add_argument("--csv", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_intraday.csv"))
    ap.add_argument("--out-text", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_post_intraday.txt"))
    args = ap.parse_args()

    df = read_intraday(Path(args.csv))
    if df.empty:
        Path(args.out_text).write_text(f"{args.index_key.upper()} intraday: (no data)\n", encoding="utf-8")
        return

    # 当日データ
    day = df["ts"].dt.floor("D").iloc[-1]
    df_day = df[df["ts"].dt.floor("D") == day]
    if df_day.empty:
        Path(args.out_text).write_text(f"{args.index_key.upper()} intraday: (no data)\n", encoding="utf-8")
        return

    base, basis_note = choose_baseline(df_day)
    first_ts = df_day.iloc[0]["ts"]
    last_ts = df_day.iloc[-1]["ts"]
    last_val = float(df_day.iloc[-1]["val"])

    delta_level = last_val - float(base)
    pct_val = percent_change(base, last_val)

    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = f"{delta_level:+.6f}"

    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={first_ts}->{last_ts})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
