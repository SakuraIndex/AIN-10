#!/usr/bin/env python3
"""
1日の騰落率（A%）を計算して、docs/outputs に
- <index>_post_intraday.txt         … 投稿用1行
- <index>_stats.json                … pct_1d を反映（他フィールドは触らない or 無ければ最小限で作成）
を書き込みます。

寄り値は「当日 09:00 以降で最初のティック」。無ければ当日の最初のティック。
値列は自動検出（候補名 or 最初の数値列）。
"""

from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import Optional, List

import pandas as pd


# ========= ユーティリティ =========

def first_existing(path_candidates: List[Path]) -> Optional[Path]:
    for p in path_candidates:
        if p.exists():
            return p
    return None


def detect_time_col(df: pd.DataFrame) -> str:
    """日時列を自動検出。最初に datetime64 になり得る列を返す。"""
    # 既知の候補
    for c in ["ts", "time", "timestamp", "date", "datetime", "Datetime"]:
        if c in df.columns:
            return c
    # 最初に datetime に変換できる列
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    raise ValueError("日時列を検出できませんでした。")


def detect_value_col(df: pd.DataFrame, index_key: str, time_col: str) -> str:
    """値列を自動検出。候補名の優先順→最初の数値列。"""
    candidates = [
        "value", "y", "index", "score", "close", "price",
        index_key, index_key.replace("-", ""), index_key.replace("_", ""),
        index_key.upper(), index_key.upper().replace("-", "").replace("_", ""),
        "AIN10", "AIN-10",
    ]
    # 先に候補名
    for c in candidates:
        if c in df.columns:
            try:
                pd.to_numeric(df[c])
                return c
            except Exception:
                pass
    # 次に「数値に変換できる列」を探索（ただし time_col は除外）
    for c in df.columns:
        if c == time_col:
            continue
        try:
            pd.to_numeric(df[c])
            return c
        except Exception:
            pass
    raise ValueError("値列を検出できませんでした。")


def pick_open_row(day_df: pd.DataFrame, tcol: str) -> int:
    """
    当日 09:00 以降で最初のティックの index を返す。
    見つからなければ当日の最初の行（昇順で 0）を返す。
    """
    # 09:00 以上の最初
    after_9 = day_df[day_df[tcol].dt.time >= pd.to_datetime("09:00").time()]
    if len(after_9) > 0:
        return after_9.index[0]
    return day_df.index[0]


# ========= ここからメイン =========

def main() -> None:
    index_key = os.getenv("INDEX_KEY", "ain10").strip()
    out_dir = Path("docs/outputs")

    # CSV の候補（将来フォーマット差異に備えて複数候補）
    csv_path = first_existing([
        out_dir / f"{index_key}_1d.csv",
        out_dir / f"{index_key}.csv",
    ])
    if csv_path is None:
        raise FileNotFoundError("1日足CSVが見つかりません: docs/outputs/<index>_1d.csv")

    df = pd.read_csv(csv_path)
    # 日時列・値列を検出して正規化
    tcol = detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).copy()

    vcol = detect_value_col(df, index_key=index_key, time_col=tcol)
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=[vcol]).copy()

    # 当日（= 最終日）の抽出 & 時刻昇順
    last_date = df[tcol].dt.date.max()
    day_df = df[df[tcol].dt.date == last_date].copy()
    if day_df.empty:
        raise ValueError("当日のデータ行が見つかりません。")

    day_df = day_df.sort_values(tcol).reset_index(drop=True)

    # open と latest を取得
    open_idx = pick_open_row(day_df, tcol)
    open_val = float(day_df.loc[open_idx, vcol])
    latest_val = float(day_df.loc[day_df.index[-1], vcol])

    # パーセント計算（0除算防止）
    pct = float("nan")
    if open_val != 0 and math.isfinite(open_val):
        pct = (latest_val / open_val - 1.0) * 100.0

    # 表示テキスト
    valid_str = f"{day_df.loc[0, tcol].strftime('%Y-%m-%d %H:%M')} -> {day_df.loc[day_df.index[-1], tcol].strftime('%Y-%m-%d %H:%M')}"
    if math.isfinite(pct):
        pct_str = f"{pct:+.2f}%"
        delta_level = latest_val - open_val
        delta_str = f"{delta_level:+.6f}"
    else:
        pct_str = "N/A"
        delta_str = "N/A"

    line = (
        f"{index_key.upper()} 1d: "
        f"Δ={delta_str} (level) "
        f"A%={pct_str} (basis=open valid={valid_str} open -> latest)"
    )

    # 投稿用テキスト
    (out_dir / f"{index_key}_post_intraday.txt").write_text(line + "\n", encoding="utf-8")

    # stats.json の更新（pct_1d だけを上書き、他キーは維持）
    stats_path = out_dir / f"{index_key}_stats.json"
    stats = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}

    stats.setdefault("index_key", index_key)
    stats["pct_1d"] = float(f"{pct:.6f}") if math.isfinite(pct) else None
    # basis 情報だけ残す（可視化側が読む想定なら）
    stats["basis"] = "open"
    # delta_level はこのスクリプトでは責務外にしてもOK。必要なら open 基準の差を入れても良い。
    # stats["delta_level"] = delta_level if math.isfinite(pct) else None
    # stats["scale"] = "level"

    # タイムスタンプ
    stats["updated_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
