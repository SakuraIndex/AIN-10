#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "outputs"

HISTORY_CSV = OUT / "ain10_history.csv"   # 日次の終値が入っている想定
STATS_JSON  = OUT / "ain10_stats.json"
POST_TXT    = OUT / "ain10_post_intraday.txt"

INDEX_KEY = "ain10"

def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"history csv not found: {path}")
    df = pd.read_csv(path)
    # 列名のゆらぎを吸収
    # 期待: date(またはtimestamp), close(またはprice/value)
    cols = {c.lower(): c for c in df.columns}
    date_col = None
    for k in ("date", "timestamp"):
        if k in cols:
            date_col = cols[k]
            break
    value_col = None
    for k in ("close", "price", "value"):
        if k in cols:
            value_col = cols[k]
            break
    if date_col is None or value_col is None:
        raise ValueError(
            f"Required columns not found. Need one of date/timestamp and close/price/value. got: {list(df.columns)}"
        )
    df = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "close"})
    # 並び替え＆欠損除去
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["close"])
    return df

def compute_pct_1d(df: pd.DataFrame):
    if len(df) < 2:
        return None, None, None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_close = float(prev["close"])
    last_close = float(last["close"])
    if prev_close == 0:
        return None, prev["date"].date().isoformat(), last["date"].date().isoformat()
    pct = (last_close - prev_close) / prev_close * 100.0
    return pct, prev["date"].date().isoformat(), last["date"].date().isoformat()

def update_stats_json(pct_1d: float | None):
    # 既存のキーは維持して追記/更新
    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": None,
        "delta_level": None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if STATS_JSON.exists():
        try:
            existing = json.loads(STATS_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                stats |= existing
        except Exception:
            pass
    stats["index_key"] = INDEX_KEY
    stats["pct_1d"] = round(pct_1d, 6) if pct_1d is not None else None
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

def write_post_txt(pct_1d: float | None, prev_d: str | None, last_d: str | None):
    if pct_1d is None or prev_d is None or last_d is None:
        line = "AIN10 1d: A%=N/A (basis n/a)"
    else:
        sign = "+" if pct_1d >= 0 else ""
        line = f"AIN10 1d: A%={sign}{pct_1d:.2f}% (basis prev-close {prev_d}->{last_d})"
    POST_TXT.write_text(line + "\n", encoding="utf-8")

def main():
    try:
        df = load_history(HISTORY_CSV)
        pct_1d, prev_d, last_d = compute_pct_1d(df)
        update_stats_json(pct_1d)
        write_post_txt(pct_1d, prev_d, last_d)
        print("ok")
    except Exception as e:
        # 失敗時は N/A を残して終了（CI を落とさない）
        update_stats_json(None)
        write_post_txt(None, None, None)
        print(f"warn: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
