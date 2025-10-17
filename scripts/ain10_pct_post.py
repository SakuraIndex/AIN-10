#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute 1-day percentage change for X posting and update stats.

- 入力: docs/outputs/{index}_1d.csv（Datetime + level 列）
- ロジック:
    基本は「前営業日の最終値」を基準（basis='prev_close'）。
    もし前日データが無い場合は「当日の最初の値」を基準（basis='open'）。
- 出力:
    docs/outputs/{index}_post_intraday.txt   ← Xに貼る短文
    docs/outputs/{index}_stats.json          ← pct_1d / delta_level / basis を更新（scale=level）

環境変数:
    INDEX_KEY: 例 'ain10'
"""

from __future__ import annotations
import os
import json
import pathlib as p
import pandas as pd
import numpy as np

OUT_DIR = p.Path("docs/outputs")

def _detect_cols(df: pd.DataFrame):
    time_cands = ["Datetime","datetime","timestamp","time","date","Date"]
    val_cands  = ["value","level","score","close","price","AIN-10"]

    tcol = next((c for c in time_cands if c in df.columns), None)
    if tcol is None:
        # indexがDatetimeIndexなら救済
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Datetime"})
            tcol = "Datetime"
        else:
            # 先頭で時刻に変換できる列を探す
            for c in df.columns:
                try:
                    pd.to_datetime(df[c])
                    tcol = c
                    break
                except Exception:
                    pass
    if tcol is None:
        raise ValueError(f"time-like column not found. columns={list(df.columns)}")

    vcol = next((c for c in val_cands if c in df.columns), None)
    if vcol is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError("value-like column not found.")
        vcol = numeric[0]

    return df, tcol, vcol

def load_1d(index_key: str) -> pd.DataFrame:
    csv_path = OUT_DIR / f"{index_key}_1d.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df, tcol, vcol = _detect_cols(df)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    df = df[[tcol, vcol]].rename(columns={tcol: "ts", vcol: "level"})
    return df

def compute_pct(df: pd.DataFrame):
    # 最新日の切り出し
    df["date"] = df["ts"].dt.date
    cur_date = df["date"].iloc[-1]
    day_df = df[df["date"] == cur_date]
    latest_ts  = day_df["ts"].iloc[-1]
    latest_val = float(day_df["level"].iloc[-1])

    # デフォルト：前日終値基準
    prev_dates = sorted(df["date"].unique())
    prev_dates = [d for d in prev_dates if d < cur_date]
    if prev_dates:
        prev_day = prev_dates[-1]
        prev_close = float(df[df["date"] == prev_day]["level"].iloc[-1])
        basis_val = prev_close
        basis = "prev_close"
        valid_note = f"{prev_day} -> {cur_date}"
    else:
        # 前日がないなら当日始値
        basis_val = float(day_df["level"].iloc[0])
        basis = "open"
        valid_note = f"{cur_date} open -> latest"

    delta_level = latest_val - basis_val
    # 0割り防止：絶対値が極小なら%は0扱い
    if abs(basis_val) < 1e-12:
        pct = 0.0
    else:
        pct = (latest_val - basis_val) / abs(basis_val) * 100.0

    return delta_level, pct, basis, valid_note, latest_ts

def write_post(index_key: str, delta_level: float, pct: float,
               basis: str, valid_note: str):
    txt = (
        f"{index_key.upper()} 1d: Δ={delta_level:+.6f} (level) "
        f"A%={pct:+.2f}% (basis={basis} valid={valid_note})"
    )
    (OUT_DIR / f"{index_key}_post_intraday.txt").write_text(txt, encoding="utf-8")

def update_stats(index_key: str, delta_level: float, pct: float, basis: str):
    # 既存 stats があれば読み、無ければ新規
    stats_path = OUT_DIR / f"{index_key}_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    else:
        stats = {}

    stats.update({
        "index_key": index_key,
        "pct_1d": pct,
        "delta_level": delta_level,
        "scale": "level",
        "basis": basis,
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

def main():
    index_key = os.environ.get("INDEX_KEY", "").strip()
    if not index_key:
        raise SystemExit("Env INDEX_KEY is required (e.g. ain10)")

    df = load_1d(index_key)
    delta_level, pct, basis, valid_note, latest_ts = compute_pct(df)
    write_post(index_key, delta_level, pct, basis, valid_note)
    update_stats(index_key, delta_level, pct, basis)

if __name__ == "__main__":
    main()
