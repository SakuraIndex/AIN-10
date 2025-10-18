#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
import pandas as pd

def iso_now() -> str:
    """UTC ISO8601（Z付き）"""
    ts = pd.Timestamp.utcnow()
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {path}")
    # 先頭2列を ts/val に寄せるのは 1d 用。intraday は後で列探索。
    return df

def read_1d(csv_1d: Path) -> tuple[float | None, str]:
    """1d の Δlevel と有効区間メモを返す"""
    df = load_csv(csv_1d).copy()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        return None, "n/a"
    first = float(df.iloc[0]["val"])
    last  = float(df.iloc[-1]["val"])
    delta = last - first
    valid = f"{df.iloc[0]['ts']}->{df.iloc[-1]['ts']}"
    return delta, valid

def find_pct_column(df: pd.DataFrame) -> str | None:
    """intraday から % 列を見つける。候補: 'A%', 'pct', 'percent'（大小区別なし/記号含む）"""
    candidates = [c for c in df.columns]
    for c in candidates:
        name = str(c).strip()
        if re.fullmatch(r"(?i)a%|pct|percent|pct_\d*d?", name.replace(" ", "")):
            return c
    # 記号の含まれ方違いにもゆるめに対応
    for c in candidates:
        n = str(c).lower()
        if "a%" in n or "pct" in n or "percent" in n:
            return c
    return None

def read_intraday_pct(csv_intraday: Path) -> tuple[float | None, str]:
    """
    intraday の % 列（open→latest の累積％）をそのまま採用。
    見つからなければ None を返す。
    """
    df = load_csv(csv_intraday).copy()
    # タイムスタンプ列を推定（先頭列が日時の前提）
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    pct_col = find_pct_column(df)
    if pct_col is None:
        return None, "no_pct_col"
    # 最後に有効な％
    last_valid = pd.to_numeric(df[pct_col], errors="coerce").dropna()
    if last_valid.empty:
        return None, "pct_all_nan"
    pct_last = float(last_valid.iloc[-1])
    return pct_last, f"from:{pct_col}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv-1d", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--csv-intraday", required=True, help="docs/outputs/*_intraday.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    # 1) Δlevel（1dから）
    delta_level, valid_note = read_1d(Path(args.csv_1d))

    # 2) 騰落率 %（intraday の % 列をそのまま）
    pct_val, basis_note = read_intraday_pct(Path(args.csv_intraday))

    # --- TXT 出力
    pct_str   = "N/A" if pct_val   is None else f"{pct_val:+.2f}%"
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
