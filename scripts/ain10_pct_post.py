# scripts/ain10_pct_post.py
# -*- coding: utf-8 -*-
"""
AIN-10 ほか共通: 1日の騰落率を robust に計算して
docs/outputs/{key}_stats.json と {key}_post_intraday.txt を更新する。
- 優先: intraday の [今日の最初値, 最新値]
- 代替: history の [前営業日終値, 最新終値]
- 未取得: N/A
"""

from __future__ import annotations
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
TZ = timezone.utc  # すべてUTCで記録（表示の都合で）

# 候補にする列名（先に見つかったものを使う）
VALUE_CANDIDATES = [
    # 価格系（最優先）
    "price", "close", "Close", "last", "adj_close",
    # スコア/レベル系（やむなく代替。当日比%の概算として扱う）
    "value", "y", "index", "score", "AIN-10", KEY.upper(), KEY,
]

TS_CANDIDATES = ["ts", "time", "timestamp", "date", "datetime", "Datetime"]

def _now_iso() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%SZ")

def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # タイムスタンプ列の標準化
    for c in TS_CANDIDATES:
        if c in df.columns:
            df["_ts"] = pd.to_datetime(df[c], errors="coerce", utc=True)
            break
    else:
        df["_ts"] = pd.NaT

    # 値段/値列の特定
    val_col = None
    for c in VALUE_CANDIDATES:
        if c in df.columns:
            val_col = c
            break
    # 数値の列が1つしかない場合、それを採用（列名不明対策）
    if val_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            val_col = num_cols[0]

    if val_col is None:
        return None

    # 数値化
    df["_v"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["_v"]).copy()
    if df.empty:
        return None

    # ソート（時系列）
    if df["_ts"].notna().any():
        df = df.sort_values("_ts")
    return df

def _first_last_today(df: pd.DataFrame) -> tuple[float | None, float | None, str]:
    """UTC日付で今日の first/last を返す。basis 文字列も返す。"""
    if "_ts" not in df.columns or df["_ts"].isna().all():
        return None, None, "n/a"

    today = datetime.now(TZ).date()
    dft = df[df["_ts"].dt.date == today]
    if dft.empty:
        return None, None, "n/a"

    first = float(dft.iloc[0]["_v"])
    last = float(dft.iloc[-1]["_v"])
    valid = f"{dft.iloc[0]['_ts'].strftime('%Y-%m-%d %H:%M')} -> latest"
    return first, last, f"open (valid={valid})"

def _prev_close_vs_last(df: pd.DataFrame) -> tuple[float | None, float | None, str]:
    """前営業日終値と最新行で計算。時系列が無ければ全体の先頭/末尾。"""
    if "_ts" in df.columns and df["_ts"].notna().any():
        df = df.sort_values("_ts")
    # 同一日ごとに末尾＝終値を採り、直近2日を使う
    if "_ts" in df.columns and df["_ts"].notna().any():
        dd = df.copy()
        dd["date"] = dd["_ts"].dt.date
        last_by_day = dd.groupby("date").tail(1)
        if len(last_by_day) >= 2:
            prev = float(last_by_day.iloc[-2]["_v"])
            last = float(last_by_day.iloc[-1]["_v"])
            valid = f"{last_by_day.iloc[-2]['_ts'].strftime('%Y-%m-%d')} close -> latest"
            return prev, last, f"prev_close (valid={valid})"

    # 仕方なく全体先頭/末尾
    prev = float(df.iloc[0]["_v"])
    last = float(df.iloc[-1]["_v"])
    return prev, last, "prev_any"

def _pct(a: float, b: float) -> float | None:
    # (b - a) / a * 100
    if a is None or b is None:
        return None
    if a == 0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    return (b - a) / a * 100.0

def compute_pct_1d() -> tuple[float | None, str]:
    """
    returns: (pct, basis)
    """
    intraday = _load_csv(OUT_DIR / f"{KEY}_intraday.csv")
    if intraday is not None:
        a, b, basis = _first_last_today(intraday)
        p = _pct(a, b)
        if p is not None:
            return p, basis

    # 代替：ヒストリー（前日終値→最新終値）
    history = _load_csv(OUT_DIR / f"{KEY}_history.csv")
    if history is not None:
        a, b, basis = _prev_close_vs_last(history)
        p = _pct(a, b)
        if p is not None:
            return p, basis

    return None, "n/a"

def update_files(pct: float | None, basis: str):
    # stats.json
    stats_path = OUT_DIR / f"{KEY}_stats.json"
    stats = {
        "index_key": KEY,
        "pct_1d": None if pct is None else float(pct),
        "delta_level": None,  # レベルΔは別処理（長期チャート側で管理）
        "scale": "level",
        "basis": basis,
        "updated_at": _now_iso(),
    }
    try:
        stats_path.write_text(json.dumps(stats, ensure_ascii=False))
    except Exception as e:
        print("write stats error:", e)

    # post_intraday.txt
    post_path = OUT_DIR / f"{KEY}_post_intraday.txt"
    if pct is None:
        pct_str = "N/A"
    else:
        sign = "+" if pct >= 0 else ""
        pct_str = f"{sign}{pct:.2f}%"
    line = f"{KEY.upper()} 1d: Δ=N/A (level) A%={pct_str} (basis={basis})\n"
    try:
        post_path.write_text(line)
    except Exception as e:
        print("write post error:", e)

def main():
    pct, basis = compute_pct_1d()
    update_files(pct, basis)

if __name__ == "__main__":
    main()
