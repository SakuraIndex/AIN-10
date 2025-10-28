#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_TZ = os.environ.get("MARKET_TZ", "America/New_York")
SESSION_START = os.environ.get("SESSION_START", "09:30")
SESSION_END   = os.environ.get("SESSION_END",   "16:00")

def diag(msg): print(f"[long] {msg}", flush=True)

def apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(110)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists(): return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2: return pd.DataFrame()
    df = df.rename(columns={df.columns[0]:"date", df.columns[1]:"value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    return df

def slice_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty: return df
    last = df["date"].max()
    w = df[df["date"] >= (last - pd.Timedelta(days=days))].copy()
    return w

def plot_level(dfp: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Level (index)", labelpad=10)

    if len(dfp) < 2:
        # 点だけでも出す + 注記
        if not dfp.empty:
            ax.plot(dfp["date"].values, dfp["value"].values, marker="o", linewidth=0)
        ax.text(0.03, 0.92, "Insufficient history (need ≥ 2 days)", transform=ax.transAxes, color="#aaaaaa")
    else:
        ax.plot(dfp["date"].values, dfp["value"].values, linewidth=2.6, color="#ff615a")

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    diag(f"WROTE: {out_png}")

def main():
    hist = OUT_DIR / f"{INDEX_KEY}_history.csv"
    df = load_history(hist)
    if df.empty:
        diag("history empty; nothing to render.")
        return

    spans = {
        "7d": 7,
        "1m": 30,
        "1y": 365,
    }
    for span, days in spans.items():
        w = slice_window(df, days)
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_level(w, f"{INDEX_KEY.upper()} ({span} level)", out_png)

if __name__ == "__main__":
    diag(f"INDEX_KEY={INDEX_KEY}  TZ={MARKET_TZ}  SESSION={SESSION_START}-{SESSION_END}")
    main()
