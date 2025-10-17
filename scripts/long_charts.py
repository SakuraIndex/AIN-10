# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
AIN 系の長期チャート(1d/7d/1m/1y)を描画し PNG を出力するスクリプト。
- 背景: 黒 (dark_background)
- 線色: 上昇=赤, 下落=青, 横ばい=グレー
- チャート上に%は出さない（タイトル等へも出さない）
- stats.json は "pct_1d": null を維持、"delta_level" のみ記録、"basis": "n/a"
- 引数なしで動作（環境変数 INDEX_KEY を使用）。ワークフローからそのまま呼び出せます。

CSVの想定:
  docs/outputs/{index_key}_1d.csv
  docs/outputs/{index_key}_7d.csv
  docs/outputs/{index_key}_1m.csv
  docs/outputs/{index_key}_1y.csv

列名は柔軟に解決する:
  - 時刻列候補: ["ts", "time", "timestamp", "date", "Datetime", "datetime"]
  - 値列候補: ["value", "level", index_key] もしくは「最初の数値列」
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI不要環境
import matplotlib.pyplot as plt


OUTPUT_DIR = "docs/outputs"
TIME_COL_CANDIDATES = ["ts", "time", "timestamp", "date", "Datetime", "datetime"]
VALUE_COL_CANDIDATES = ["value", "level"]  # index_key は後で追加
PNG_TITLE_FMT = "{index_key_upper} ({label})"

SERIES = [
    ("1d", "1d"),
    ("7d", "7d"),
    ("1m", "1m"),
    ("1y", "1y"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pick_time_col(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    # Fallback: 最も「時刻っぽい」列を選ぶ（型/名前で推測）
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    # どうしても無い場合は先頭列
    return df.columns[0]


def pick_value_col(df: pd.DataFrame, index_key: str) -> str:
    candidates = VALUE_COL_CANDIDATES + [index_key, index_key.upper(), index_key.replace("_", "-").upper()]
    for c in candidates:
        if c in df.columns:
            return c
    # 最初に見つかった「数値列」を返す
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    # どうしても見つからない場合は2列目を仮定
    return df.columns[min(1, len(df.columns) - 1)]


def load_series(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    tcol = pick_time_col(df)
    vcol = pick_value_col(df, index_key)

    # 時刻列を datetime 化
    df["ts"] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    # 値列を float 化
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")

    # 欠損除去・並び替え
    df = df.dropna(subset=["ts", "value"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{csv_path}: 有効データがありません (ts/value が空)")

    return df[["ts", "value"]]


def line_color_for_delta(delta_level: float) -> str:
    if delta_level > 0:
        return "#ff6b6b"  # 上昇=赤
    if delta_level < 0:
        return "#4da3ff"  # 下落=青
    return "#aaaaaa"      # 横ばい=グレー


def render_level_chart(csv_path: str, png_path: str, title: str, index_key: str) -> Tuple[float, Tuple[pd.Timestamp, pd.Timestamp]]:
    df = load_series(csv_path, index_key)

    # Δ(level)
    delta_level = float(np.round(df["value"].iloc[-1] - df["value"].iloc[0], 6))
    color = line_color_for_delta(delta_level)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

    ax.plot(df["ts"], df["value"], linewidth=2.2, color=color)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.grid(True, alpha=0.28)

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)

    valid = (df["ts"].iloc[0], df["ts"].iloc[-1])
    return delta_level, valid


def write_stats_json(index_key: str, delta_level: Optional[float]) -> None:
    """
    グローバルな stats JSON を書き出し。
    - pct_1d は常に null（レベルでは%表示しない）
    - delta_level は直近 1d の Δ(level)。計算できない/未生成なら null。
    - basis は "n/a"
    """
    out = {
        "index_key": index_key,
        "pct_1d": None,
        "delta_level": (float(delta_level) if delta_level is not None else None),
        "scale": "level",
        "basis": "n/a",
        "updated_at": _utc_now_iso(),
    }
    out_path = os.path.join(OUTPUT_DIR, f"{index_key}_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[stats] wrote: {out_path}")


def write_post_marker(index_key: str) -> None:
    """
    X 投稿などで拾うプレーンテキスト（レベルでは%を出さない）。
    """
    text = f"{index_key.upper()} 1d: A%=N/A (basis n/a)"
    out_path = os.path.join(OUTPUT_DIR, f"{index_key}_post_intraday.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[post] wrote: {out_path} -> {text}")


def main() -> None:
    index_key = os.environ.get("INDEX_KEY", "").strip().lower()
    if not index_key:
        raise SystemExit("環境変数 INDEX_KEY が未設定です")

    # 各期間の CSV と PNG
    # 見つかったものだけ描画（無ければスキップ）
    last_delta_1d: Optional[float] = None

    for label, suffix in SERIES:
        csv_path = os.path.join(OUTPUT_DIR, f"{index_key}_{suffix}.csv")
        png_path = os.path.join(OUTPUT_DIR, f"{index_key}_{suffix}.png")

        if not os.path.exists(csv_path):
            print(f"[skip] CSV not found: {csv_path}")
            continue

        title = PNG_TITLE_FMT.format(index_key_upper=index_key.upper(), label=label)
        try:
            delta_level, valid = render_level_chart(csv_path, png_path, title, index_key)
            print(f"[ok] {suffix} -> Δ(level)={delta_level:.6f} valid={valid[0]}->{valid[1]}")
            if suffix == "1d":
                last_delta_1d = delta_level
        except Exception as e:
            print(f"[error] render {suffix}: {e}")

    # マーカー類の出力（%はN/Aの方針）
    write_stats_json(index_key, last_delta_1d)
    write_post_marker(index_key)


if __name__ == "__main__":
    main()
