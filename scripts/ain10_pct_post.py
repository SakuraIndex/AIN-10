#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
import pandas as pd
from typing import Optional

EPS = 1e-6          # 非ゼロ判定のしきい値
SAFE_WINDOW_MIN = 5 # 市場オープンから最初の探索オフセット(分)
MAX_WINDOW_MIN  = 30# この分数までに非ゼロが見つからなければ全体から探す

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

def find_nonzero_open(df: pd.DataFrame) -> tuple[Optional[float], str]:
    """
    オープン(最初の行)直後の“実質ゼロでない”最初の値を探す。
    - オープン時刻 + SAFE_WINDOW_MIN 〜 +MAX_WINDOW_MIN の範囲を優先
    - 見つからなければ全体で最初の非ゼロ
    戻り値: (値, basis_note)
    """
    if df.empty:
        return None, "n/a"

    t0 = df.iloc[0]["ts"]
    lower = t0 + pd.Timedelta(minutes=SAFE_WINDOW_MIN)
    upper = t0 + pd.Timedelta(minutes=MAX_WINDOW_MIN)

    m = (df["ts"] >= lower) & (df["ts"] <= upper) & (df["val"].abs() >= EPS)
    if m.any():
        row = df[m].iloc[0]
        when = pd.to_datetime(row["ts"])
        return float(row["val"]), f"open(nonzero@{when.strftime('%H:%M')})"

    m2 = (df["val"].abs() >= EPS)
    if m2.any():
        row = df[m2].iloc[0]
        when = pd.to_datetime(row["ts"])
        return float(row["val"]), f"open(first_nonzero@{when.strftime('%H:%M')})"

    return None, "open(all_zero)"

def percent_change(first: float, last: float) -> Optional[float]:
    """NaNや0割防止付きの単純騰落率(%単位)"""
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
    # ← 互換目的で受け取って無視する（ワークフローが渡しても落ちないように）
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

        base_val, base_note = find_nonzero_open(df)
        basis_note = base_note if args.basis == "open" else "prev_close"

        delta_level = last_val - float(first_row["val"])
        pct_val = percent_change(base_val, last_val)

    # --- TXT 出力
    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
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
