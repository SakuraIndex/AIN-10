#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docs/outputs/* の 1d/7d/1m/1y CSV を読み、黒ベースの画像(PNG)を出力します。
- 背景: 濃いダーク (#0e0f12)
- グリッド: さりげないグレー (#2a2d34, alpha 0.28)
- 線: インデックス用の赤 (#ff6b6b) 1.8pt
- 余白の白フチが出ないよう savefig(facecolor=fig.get_facecolor(), bbox_inches="tight")
"""

import os
import sys
import math
from datetime import timezone
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()
OUT_DIR = "docs/outputs"

# -------- ダークテーマ（見た目を元の黒基調へ） --------
BG = "#0e0f12"
FG = "#e6e6e6"
GRID = "#2a2d34"
LINE = "#ff6b6b"

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.28,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "font.size": 12,
})

def _pick_time_col(df: pd.DataFrame):
    cand = ["Datetime","datetime","timestamp","time","date","ts"]
    for c in cand:
        if c in df.columns:
            return c
    # 1列目が時刻の CSV にも雑に対応
    return df.columns[0]

def _pick_value_col(df: pd.DataFrame):
    # インデックス列名の候補
    k = INDEX_KEY
    cands = [
        k, k.upper(), k.title(), k.replace("_","-").upper(),
        "value", "Value", "index", "score", "close", "price", "y",
        "AIN-10", "AIN10"
    ]
    for c in cands:
        if c in df.columns:
            return c
    # 最後の列を値とみなす（2列 CSV など）
    return df.columns[-1]

def _load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    tcol = _pick_time_col(df)
    vcol = _pick_value_col(df)
    df = df[[tcol, vcol]].copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    # 値は数値化
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    return df.rename(columns={tcol: "ts", vcol: "val"}).reset_index(drop=True)

def _title_case(label: str) -> str:
    # AIN-10 表記などをできるだけ崩さないように
    if label.lower() == "ain10" or label.lower() == "ain_10" or label.lower() == "ain-10":
        return "AIN10"
    return label.upper()

def _format_xaxis(ax):
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%Y-%m-%d", tz=timezone.utc))
    for spine in ax.spines.values():
        spine.set_color(FG)

def _plot_one(csv_path: str, out_png: str, title_suffix: str):
    if not os.path.exists(csv_path):
        return
    df = _load_series(csv_path)
    if df.empty or df["val"].dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.plot(df["ts"], df["val"], color=LINE, linewidth=1.8, solid_capstyle="round")

    ax.set_title(f"{_title_case(INDEX_KEY)} ({title_suffix})", color=FG, fontsize=18, pad=12, weight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    _format_xaxis(ax)

    # 余白を控えめに、黒縁の白抜け防止
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = [
        (f"{OUT_DIR}/{INDEX_KEY}_1d.csv", f"{OUT_DIR}/{INDEX_KEY}_1d.png", "1d"),
        (f"{OUT_DIR}/{INDEX_KEY}_7d.csv", f"{OUT_DIR}/{INDEX_KEY}_7d.png", "7d"),
        (f"{OUT_DIR}/{INDEX_KEY}_1m.csv", f"{OUT_DIR}/{INDEX_KEY}_1m.png", "1m"),
        (f"{OUT_DIR}/{INDEX_KEY}_1y.csv", f"{OUT_DIR}/{INDEX_KEY}_1y.png", "1y"),
    ]
    for csv_path, out_png, label in pairs:
        _plot_one(csv_path, out_png, label)

if __name__ == "__main__":
    main()
