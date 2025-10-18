#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

def iso_now() -> str:
    # tz-aware/naive どちらでもOKな安全実装
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # basis はメモだけ（動作は差分固定）
    ap.add_argument("--basis", default="stable@10:00")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    delta_pp = None
    valid_note = "n/a"

    if not df.empty:
        # 当日最初と最後（10:00 アンカーが別CSVに無ければ、当日先頭をアンカー扱い）
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])

        # ここが重要：比率ではなく「差分（pp）」のみ
        delta_pp = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

    # TXT 出力（A%= は percentage **points** の意味）
    a_str = "N/A" if delta_pp is None else f"{delta_pp:+.2f}pp"
    d_str = "N/A" if delta_pp is None else f"{delta_pp:+.6f}"  # 生のレベル差（pp）
    text = (
        f"{args.index_key.upper()} 1d: Δ={d_str} (level) "
        f"A%={a_str} (basis={args.basis} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # JSON 出力: pct_1d は「ppとしての値」をそのまま入れる
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if delta_pp is None else float(delta_pp),
        "delta_level": None if delta_pp is None else float(delta_pp),
        "scale": "level",      # level（＝pp）であることを明示
        "basis": args.basis,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
