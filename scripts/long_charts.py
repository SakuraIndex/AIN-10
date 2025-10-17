# scripts/long_charts.py
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
import os

TIME_CANDIDATES = ["ts","time","timestamp","date","datetime","Datetime"]

DARK_BG   = "#0b0f14"
AX_BG     = "#0b0f14"
GRID_COL  = "#9aa4b126"  # 薄い
EDGE_COL  = "#222831"
TICK_COL  = "#e6e6e6"
TITLE_COL = "#ffffff"
LINE_COL  = "#ff6b6b"

def load_series(csv_path: str):
    df = pd.read_csv(csv_path)
    time_col = None
    for c in TIME_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        raise ValueError("value column not found.")
    vcol = value_cols[-1]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    y = pd.to_numeric(df[vcol], errors="coerce")
    df = df.assign(_y=y).dropna(subset=["_y"])
    x = df[time_col].dt.tz_convert(None) if df[time_col].dt.tz is not None else df[time_col]
    return x, df["_y"]

def save_chart(x, y, title: str, out_png: str):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    # 背景
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)

    # 線
    ax.plot(x, y, LINE_COL, lw=2.2)

    # 体裁
    ax.set_title(title, color=TITLE_COL, fontsize=18, pad=14)
    ax.set_xlabel("Time", color=TICK_COL)
    ax.set_ylabel("Index (level)", color=TICK_COL)
    for spine in ax.spines.values():
        spine.set_color(EDGE_COL)
    ax.tick_params(colors=TICK_COL, labelsize=10)
    ax.grid(True, which="major", color=GRID_COL, linewidth=1)
    ax.grid(True, which="minor", color=GRID_COL, linewidth=0.5)
    ax.margins(x=0.01, y=0.05)

    # x軸フォーマッタ
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.minorticks_on()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, facecolor=DARK_BG, edgecolor=DARK_BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def do_one(index_key: str, horizon: str):
    csv = f"docs/outputs/{index_key}_{horizon}.csv"
    png = f"docs/outputs/{index_key}_{horizon}.png"
    if not Path(csv).exists():
        return
    x, y = load_series(csv)
    title = f"{index_key.upper()} ({horizon})"
    save_chart(x, y, title, png)

def main():
    index_key = os.environ.get("INDEX_KEY", "ain10")
    for h in ["1d","7d","1m","1y"]:
        do_one(index_key, h)

if __name__ == "__main__":
    main()
