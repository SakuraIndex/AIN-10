#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd

INDEX_KEY_DEFAULT = "ain10"
OUT_DIR = Path("docs/outputs")

# 騰落率ロジック（long_charts / ain10_pct_post と整合）
EPS = 5.0
CLAMP_PCT = 30.0

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
        denom = max(abs(float(first)), abs(float(last)), EPS)
        pct = (float(last) - float(first)) / denom * 100.0
        if pct > CLAMP_PCT:
            pct = CLAMP_PCT
        elif pct < -CLAMP_PCT:
            pct = -CLAMP_PCT
        return pct
    except Exception:
        return None

def iso_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")

def write_all_text_variants(text: str, primary: Path):
    """
    互換のため TXT を4つの名前に出力：
      1) ain10_post_intraday.txt        （現行）
      2) ain_10_post_intraday.txt       （アンダースコア互換）
      3) ain10_intraday_post.txt        （語順互換）
      4) ain_10_intraday_post.txt       （語順＋アンダースコア互換）
    """
    variants = {
        primary.name,  # ain10_post_intraday.txt
        primary.name.replace("ain10_", "ain_10_"),
        primary.name.replace("post_intraday", "intraday_post"),
        primary.name.replace("ain10_", "ain_10_").replace("post_intraday", "intraday_post"),
    }
    for name in variants:
        (primary.parent / name).write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", default=INDEX_KEY_DEFAULT)
    ap.add_argument("--csv", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_intraday.csv"))
    ap.add_argument("--out-text", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_post_intraday.txt"))
    ap.add_argument("--out-json", default=None)  # サイト表示用（stats.jsonをintradayで上書き）
    args = ap.parse_args()

    df = read_intraday(Path(args.csv))
    if df.empty:
        text = f"{args.index_key.upper()} intraday: (no data)\n"
        write_all_text_variants(text, Path(args.out_text))
        if args.out_json:
            payload = {
                "index_key": args.index_key,
                "pct_1d": None,
                "delta_level": None,
                "scale": "percent",
                "basis": "no_data",
                "updated_at": iso_now(),
            }
            Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    # 当日データ
    day = df["ts"].dt.floor("D").iloc[-1]
    df_day = df[df["ts"].dt.floor("D") == day]
    if df_day.empty:
        text = f"{args.index_key.upper()} intraday: (no data)\n"
        write_all_text_variants(text, Path(args.out_text))
        if args.out_json:
            payload = {
                "index_key": args.index_key,
                "pct_1d": None,
                "delta_level": None,
                "scale": "percent",
                "basis": "no_pct_col",
                "updated_at": iso_now(),
            }
            Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    base, basis_note = choose_baseline(df_day)
    first_ts = df_day.iloc[0]["ts"]
    last_ts  = df_day.iloc[-1]["ts"]
    last_val = float(df_day.iloc[-1]["val"])

    delta_level = last_val - float(base)
    pct_val = percent_change(base, last_val)

    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={first_ts}->{last_ts})\n"
    )

    # TXT（4バリアント）出力
    write_all_text_variants(text, Path(args.out_text))

    # JSON（サイト表示用）を intraday で上書き
    if args.out_json:
        payload = {
            "index_key": args.index_key,
            "pct_1d": None if pct_val is None else float(pct_val),
            "delta_level": float(delta_level),
            "scale": "percent",
            "basis": basis_note,
            "updated_at": iso_now(),
        }
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
