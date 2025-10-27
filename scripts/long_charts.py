#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 long-term charts generator (1d / 7d / 1m / 1y)
- Uses *_history.csv if available (preferred)
- Falls back to per-span CSVs if not found
- Dark theme with white labels
- Clamped to ±30% change
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 5.0          # denominator lower bound
CLAMP_PCT = 30.0   # visual clamp (±30%)

# ==== theme ====
def apply_dark(ax, fig):
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

# ==== load CSV ====
def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[long] CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ==== calculations ====
def clamp(p: float) -> float:
    return max(-CLAMP_PCT, min(CLAMP_PCT, p))

def calc_pct(base: float, v: float) -> float:
    denom = max(abs(base), abs(v), EPS)
    return clamp((v - base) / denom * 100.0)

def make_pct(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df
    if span == "1d":
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day].copy()
        if d.empty:
            return d
        base = float(d.iloc[0]["val"])
        d["pct"] = d["val"].apply(lambda x: calc_pct(base, x))
        return d
    days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
    last = df["ts"].max()
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty:
        return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda x: calc_pct(base, x))
    return w

# ==== plot ====
def plot(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark(ax, fig)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(df["ts"].values, df["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[long] wrote {out_png}")

# ==== main ====
def main():
    spans = ["1d", "7d", "1m", "1y"]
    hist = OUT_DIR / f"{INDEX_KEY}_history.csv"

    # 1️⃣ history優先
    if hist.exists():
        df_all = load_csv(hist)
        if df_all.empty:
            print("[long] history empty, fallback to per-span CSVs")
        else:
            for span in spans:
                dfp = make_pct(df_all, span)
                if dfp.empty or "pct" not in dfp:
                    continue
                out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
                plot(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)
            return

    # 2️⃣ fallback: 各spanごとのCSV
    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        df = load_csv(csv)
        if df.empty:
            continue
        dfp = make_pct(df, span)
        if dfp.empty or "pct" not in dfp:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
