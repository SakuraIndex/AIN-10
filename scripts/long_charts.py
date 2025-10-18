#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_dark_theme(fig, ax):
    ax.set_facecolor("#111317")
    fig.patch.set_facecolor("#111317")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#cfd3dc", labelsize=10)
    ax.yaxis.label.set_color("#cfd3dc")
    ax.xaxis.label.set_color("#cfd3dc")
    ax.title.set_color("#e7ebf3")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {p}")
    ts, val = df.columns[:2]
    df = df.rename(columns={ts: "ts", val: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def subset_for_span(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if span == "1d":
        day = df["ts"].dt.floor("D").max()
        return df[df["ts"].dt.floor("D") == day]
    last = df["ts"].max()
    if span == "7d":
        return df[df["ts"] >= (last - pd.Timedelta(days=7))]
    if span == "1m":
        return df[df["ts"] >= (last - pd.Timedelta(days=30))]
    if span == "1y":
        return df[df["ts"] >= (last - pd.Timedelta(days=365))]
    return df

def plot_one_span(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)

    # ここがポイント：値は「pp」そのまま。%に変換しない
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (pp)", labelpad=10)

    ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    for span in ["1d", "7d", "1m", "1y"]:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue
        df = load_csv(csv)
        part = subset_for_span(df, span)
        if part.empty:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        # タイトルに basis を混ぜたい場合は ain10_stats.json を読む運用でもOK
        plot_one_span(part, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
