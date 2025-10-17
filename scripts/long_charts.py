#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黒ベースの落ち着いたチャート生成（枠線なし / grid控えめ）
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()
OUT_DIR = "docs/outputs"

BG = "#0d0e11"
FG = "#e0e0e0"
GRID = "#2d2f33"
LINE = "#ff5c5c"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": BG,  # ← 白線フレーム完全無効化
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.25,
    "font.size": 11,
})

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), df.columns[0])
    vcol = next((c for c in df.columns if c.lower() not in ["time", "date", "timestamp"]), df.columns[-1])
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol])
    return df.rename(columns={tcol:"ts", vcol:"val"}).sort_values("ts")

def plot_chart(csv_path, out_path, label):
    if not os.path.exists(csv_path):
        return
    df = load_csv(csv_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12,5), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.plot(df["ts"], df["val"], color=LINE, lw=1.8)
    ax.set_title(f"AIN10 ({label})", color=FG, fontsize=17, weight="bold", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%Y-%m-%d"))

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for span in ["1d","7d","1m","1y"]:
        plot_chart(f"{OUT_DIR}/{INDEX_KEY}_{span}.csv", f"{OUT_DIR}/{INDEX_KEY}_{span}.png", span)

if __name__ == "__main__":
    main()
