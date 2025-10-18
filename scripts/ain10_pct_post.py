#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

# --- UTC の ISO8601 (Z) を常に安全に返す ---
def iso_now() -> str:
    # tz_localize / tz_convert を使わず、最初から tz-aware を生成
    ts = pd.Timestamp.now(tz="UTC")
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
    # basis は自由文字列で受け取る（workflow 側が 'stable@10:00' などを渡してもOK）
    ap.add_argument("--basis", default="open", help="free-form note (e.g., open / prev_close / stable@10:00)")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val = None
    delta_level = None
    basis_note = args.basis   # 表示用だけに使用
    valid_note = "n/a"

    if not df.empty:
        # 1日の最初(寄り)と最後(終値相当)で計算
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])

        delta_level = last_val - first_val
        if abs(first_val) >= 1e-9:
            pct_val = (last_val - first_val) / abs(first_val) * 100.0
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

    # --- TXT 出力（X用の短文） ---
    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力（ダッシュボード/他処理向け） ---
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
