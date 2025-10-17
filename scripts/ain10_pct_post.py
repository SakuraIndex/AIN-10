#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd


def find_datetime_col(df: pd.DataFrame) -> str:
    # 先頭候補を強めに推す
    for c in df.columns:
        lc = c.lower()
        if lc in ("datetime", "timestamp", "ts", "time", "date"):
            return c
    # 見つからなければ1列目を日時として解釈
    return df.columns[0]


def find_value_col(df: pd.DataFrame) -> str:
    # 価格(終値/ラスト)候補
    prefs = ["close", "last", "price", "value", "val"]
    for p in prefs:
        for c in df.columns:
            if p in c.lower():
                return c
    # だめなら2列目(1列目は日時のはず)
    return df.columns[1]


def find_open_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "open" in c.lower():
            return c
    return None


def pct_from_intraday_open_latest(df: pd.DataFrame) -> tuple[float | None, dict]:
    """
    同一営業日の 'open' → 'latest' で % を計算する。
    戻り値: (pct_1d, meta)
    meta = {"basis": "open", "valid": "YYYY-MM-DD open -> latest"}
    """
    # 日時
    tcol = find_datetime_col(df)
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], utc=False, errors="coerce")
    df = df.dropna(subset=[tcol])
    if df.empty:
        return None, {"basis": "n/a", "valid": "n/a"}

    # 直近営業日のデータだけに絞る
    last_day = df[tcol].dt.date.max()
    day_df = df[df[tcol].dt.date == last_day].copy()
    if day_df.empty:
        return None, {"basis": "n/a", "valid": "n/a"}

    vcol = find_value_col(day_df)
    ocol = find_open_col(day_df)

    # open 値
    if ocol and ocol in day_df.columns:
        open_val = day_df[ocol].dropna().iloc[0] if day_df[ocol].notna().any() else None
    else:
        # open列が無ければ、その日の最初の価格を open とみなす
        open_val = day_df[vcol].dropna().iloc[0] if day_df[vcol].notna().any() else None

    # latest 値
    latest_val = day_df[vcol].dropna().iloc[-1] if day_df[vcol].notna().any() else None

    if open_val is None or latest_val is None:
        return None, {"basis": "n/a", "valid": "n/a"}

    # 価格系列を想定(>0)。0/負値なら安全側で N/A
    if not np.isfinite(open_val) or not np.isfinite(latest_val) or open_val <= 0:
        return None, {"basis": "n/a", "valid": "n/a"}

    pct = (latest_val / open_val - 1.0) * 100.0
    meta = {
        "basis": "open",
        "valid": f"{last_day} open -> latest",
    }
    return float(pct), meta


def main():
    ap = argparse.ArgumentParser(description="Compute 1d % for X post from intraday price CSV.")
    ap.add_argument("--index-key", required=True)
    ap.add_argument(
        "--csv",
        required=True,
        help="intraday price CSV (must contain datetime + price columns; open column optional)",
    )
    ap.add_argument("--out-json", required=True, help="path to stats json")
    ap.add_argument("--out-text", required=True, help="path to post text")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    pct_1d, meta = pct_from_intraday_open_latest(df)

    # JSON (サイト向けマーカー)
    out = {
        "index_key": args.index_key,
        "pct_1d": None if pct_1d is None else float(pct_1d),
        "delta_level": None,     # レベルは%にしない方針
        "scale": "level",
        "basis": meta.get("basis", "n/a"),
        "updated_at": pd.Timestamp.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False))

    # X 投稿テキスト
    if pct_1d is None:
        a_pct = "N/A"
    else:
        a_pct = f"{pct_1d:+.2f}%"

    line = (
        f"{args.index_key.upper()} 1d: Δ=N/A (level) "
        f"A%={a_pct} (basis={meta.get('basis','n/a')} valid={meta.get('valid','n/a')})"
    )
    Path(args.out_text).write_text(line + "\n")


if __name__ == "__main__":
    main()
