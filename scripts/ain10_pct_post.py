#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1d の intraday CSV から 1日騰落率（open→latest）を計算し、
- docs/outputs/<index>_stats.json
- docs/outputs/<index>_post_intraday.txt
を更新するユーティリティ。

CSV 列名の揺れに強く、time/timestamp/date/datetime/Datetime などを自動検出。
値列は最初の数値列を採用する（複数ある場合は index_key と一致する列名を優先）。
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np


TIME_CANDIDATES: List[str] = [
    "ts", "time", "timestamp", "date", "datetime", "Datetime"
]


def detect_time_and_value_columns(df: pd.DataFrame, index_key: str) -> Tuple[str, str]:
    # 時間列候補を探す
    time_col: Optional[str] = None
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in TIME_CANDIDATES:
        if cand in lower_cols:
            time_col = lower_cols[cand]
            break

    # なければ、datetimeにパースできる列を総当りで探す
    if time_col is None:
        for c in df.columns:
            try:
                _ = pd.to_datetime(df[c], errors="raise", utc=False)
                time_col = c
                break
            except Exception:
                pass
    if time_col is None:
        raise ValueError(f"時間列が見つかりませんでした。候補={TIME_CANDIDATES} / columns={list(df.columns)}")

    # 値列の検出：index_key と一致する列を最優先、次に最初の数値列
    value_col: Optional[str] = None
    for c in df.columns:
        if c.lower() == index_key.lower():
            value_col = c
            break
    if value_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # 時間列を除外
        numeric_cols = [c for c in numeric_cols if c != time_col]
        if not numeric_cols:
            raise ValueError(f"数値列が見つかりませんでした。columns={list(df.columns)}")
        value_col = numeric_cols[0]

    return time_col, value_col


def calc_open_latest_pct(df: pd.DataFrame, time_col: str, value_col: str) -> Tuple[Optional[float], Optional[float], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    # 時系列として整える
    s = df[[time_col, value_col]].dropna().copy()
    if s.empty:
        return None, None, None, None

    s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
    s = s.dropna(subset=[time_col])
    s = s.sort_values(time_col)

    if s.empty:
        return None, None, None, None

    open_val = s[value_col].iloc[0]
    last_val = s[value_col].iloc[-1]

    if pd.isna(open_val) or pd.isna(last_val):
        return None, None, s[time_col].min(), s[time_col].max()

    # Δ(level) と %（open→latest）
    delta_level = float(last_val) - float(open_val)
    if open_val == 0 or pd.isna(open_val):
        pct = None
    else:
        pct = (float(last_val) - float(open_val)) / abs(float(open_val)) * 100.0

    return pct, delta_level, s[time_col].min(), s[time_col].max()


def fmt_signed_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    return f"{pct:+.2f}%"


def fmt_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "N/A"
    # level は小数もありうるのでフル精度は不要。もとの出力に寄せて 6 桁程度。
    return f"{delta:+.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True, help="ex) ain10 / astra4 など")
    ap.add_argument("--csv", required=True, help="docs/outputs/<index>_1d.csv")
    ap.add_argument("--out-json", required=True, help="docs/outputs/<index>_stats.json")
    ap.add_argument("--out-text", required=True, help="docs/outputs/<index>_post_intraday.txt")
    args = ap.parse_args()

    # CSV 読み込み（エンコーディングはGitHub Actionsの標準utf-8想定）
    df = pd.read_csv(args.csv)

    # 列名検出
    time_col, value_col = detect_time_and_value_columns(df, args.index_key)

    # 計算
    pct, delta_level, tmin, tmax = calc_open_latest_pct(df, time_col, value_col)

    # JSON を書く
    nowz = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = {
        "index_key": args.index_key,
        "pct_1d": None if pct is None else round(float(pct), 6),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",          # チャートは level 表示のまま
        "basis": "open",           # % は open→latest
        "updated_at": nowz,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    # 投稿テキストを作る
    if tmin is None or tmax is None:
        valid_str = "n/a"
    else:
        # 見やすいように HH:MM だけを出してもOKだが、要望に合わせて ISO 風でも可
        valid_str = f"{tmin.strftime('%Y-%m-%d %H:%M')} -> {tmax.strftime('%Y-%m-%d %H:%M')}"

    line = (
        f"{args.index_key.upper()} 1d: "
        f"Δ={fmt_delta(delta_level)} (level) "
        f"A%={fmt_signed_pct(pct)} "
        f"(basis=open valid={valid_str})"
    )
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
