#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Long-term charts generator for INDEX (1d / 7d / 1m / 1y).
- 画像とCSVのみ出力する（post/statは一切触らない）
- CSVが無ければスキップ
- 黒ベース（デフォルト）で白枠が出ない保存方法
- 赤線（デフォルト）。環境変数で切替可:
  - THEME=dark|light
  - LINE_COLOR=#rrggbb （例: "#22c55e"）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, time


# ========= Settings =========
INDEX_KEY = os.getenv("INDEX_KEY", "ain10").lower()
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THEME = os.getenv("THEME", "dark").lower()  # "dark" | "light"
LINE_COLOR = os.getenv("LINE_COLOR", "").strip()  # empty -> theme default

# ファイル名プレフィックス
PFX = INDEX_KEY

# 読み出し対象（存在すれば描画）
SERIES = [
    ("1d", f"{PFX}_1d.csv", f"{PFX}_1d.png", f"{INDEX_KEY.upper()} (1d)"),
    ("7d", f"{PFX}_7d.csv", f"{PFX}_7d.png", f"{INDEX_KEY.upper()} (7d)"),
    ("1m", f"{PFX}_1m.csv", f"{PFX}_1m.png", f"{INDEX_KEY.upper()} (1m)"),
    ("1y", f"{PFX}_1y.csv", f"{PFX}_1y.png", f"{INDEX_KEY.upper()} (1y)"),
]

# ========= Theme =========
def theme_colors(theme: str) -> dict:
    if theme == "light":
        return dict(
            fig="#ffffff",
            ax="#ffffff",
            grid="#e5e7eb",
            label="#111827",
            spine="#6b7280",
            line= LINE_COLOR or "#ef4444",  # 赤
        )
    # dark
    return dict(
        fig="#0b0f16",
        ax="#0b0f16",
        grid="#1f2937",
        label="#e5e7eb",
        spine="#4b5563",
        line= LINE_COLOR or "#ff6b6b",  # 赤
    )


COL = theme_colors(THEME)

# ========= CSV Loader =========
def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """時間列/値列を自動特定"""
    lower_map = {c: c.lower() for c in df.columns}
    t_candidates = ["datetime", "time", "timestamp", "date", "dt"]
    v_candidates = ["value", "y", "index", "score", "close", "price"]

    # AIN-10 のような列名に対応（ハイフン含む）
    for c in df.columns:
        if c.lower().replace("-", "") in ("ain10", "ain"):
            v_candidates.insert(0, c)

    tcol = None
    for k in t_candidates:
        for c in df.columns:
            if lower_map[c] == k:
                tcol = c
                break
        if tcol:
            break
    if tcol is None:
        # 最初の列を時間列と仮定
        tcol = df.columns[0]

    vcol = None
    for k in v_candidates:
        for c in df.columns:
            if lower_map.get(c, "") == k or c == k:
                vcol = c
                break
        if vcol:
            break
    if vcol is None:
        # 2列目を値列と仮定
        vcol = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return tcol, vcol


def load_series(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    tcol, vcol = detect_cols(df)
    df = df[[tcol, vcol]].dropna().copy()
    # 時間に変換
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    df = df.dropna()

    # 値は数値化
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna()

    df = df.sort_values(by=tcol)
    df.rename(columns={tcol: "ts", vcol: "val"}, inplace=True)
    return df


# ========= Plotter =========
def plot_series(df: pd.DataFrame, title: str, out_png: Path, is_intraday: bool) -> None:
    fig = plt.figure(figsize=(12, 5), dpi=160)
    ax = fig.add_subplot(111)

    # 背景色（白枠なし）
    fig.patch.set_facecolor(COL["fig"])
    ax.set_facecolor(COL["ax"])

    ax.plot(df["ts"], df["val"], linewidth=2.2, color=COL["line"])

    # 軸・スパイン
    for sp in ax.spines.values():
        sp.set_color(COL["spine"])
    ax.tick_params(colors=COL["label"])
    ax.xaxis.set_major_locator(MaxNLocator(8))

    # 目盛/ラベル
    ax.set_title(title, color=COL["label"], fontsize=16, pad=12, weight="600")
    ax.set_ylabel("Index (level)", color=COL["label"])
    ax.set_xlabel("Time", color=COL["label"])

    # グリッド
    ax.grid(True, linestyle="-", linewidth=0.8, color=COL["grid"], alpha=0.9)

    # 余白調整（白縁が出ないよう bbox_inches は使わない）
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        dpi=160,
    )
    plt.close(fig)


# ========= Main =========
def render_one(span_key: str, csv_name: str, png_name: str, title: str):
    csv_path = OUT_DIR / csv_name
    png_path = OUT_DIR / png_name
    df = load_series(csv_path)
    if df is None or df.empty:
        print(f"[skip] {csv_name} not found or empty.")
        return
    is_intraday = (span_key == "1d")
    plot_series(df, title, png_path, is_intraday)
    print(f"[ok] {png_name} updated.")


def main():
    for span_key, csv_name, png_name, title in SERIES:
        render_one(span_key, csv_name, png_name, title)

if __name__ == "__main__":
    main()
