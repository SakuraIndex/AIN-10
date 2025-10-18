#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

EPS = 5.0         # 分母の下限
CLAMP_PCT = 30.0  # 仕上がりの上限（±30%）

def iso_now() -> str:
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

def choose_baseline(df_day: pd.DataFrame, basis: str) -> tuple[float | None, str]:
    if df_day.empty:
        return None, "no_pct_col"
    # open 優先
    if basis in ("open", "auto"):
        open_val = float(df_day.iloc[0]["val"])
        if abs(open_val) >= EPS:
            return open_val, "open"
    # 10:00 以降
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
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", choices=["open", "stable10", "auto"], default="auto")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val = None
    delta_level = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        day = df["ts"].dt.floor("D").iloc[-1]
        df_day = df[df["ts"].dt.floor("D") == day]
        if not df_day.empty:
            desired_basis = "open" if args.basis == "open" else ("stable@10:00" if args.basis == "stable10" else "auto")
            baseline, basis_note = choose_baseline(df_day, desired_basis)
            first_ts = df_day.iloc[0]["ts"]
            last_ts  = df_day.iloc[-1]["ts"]
            valid_note = f"{first_ts}->{last_ts}"
            last_val = float(df_day.iloc[-1]["val"])
            delta_level = last_val - float(baseline)
            pct_val = percent_change(baseline, last_val)

    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    Path(args.out_text).write_text(
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) A%={pct_str} (basis={basis_note} valid={valid_note})\n",
        encoding="utf-8",
    )

    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "percent",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
