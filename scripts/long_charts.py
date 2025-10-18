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

# 分母が小さすぎる時の暴発抑制（ain10_pct_post.py と同値）
EPS = 1.0

# ---- ダークテーマ ----
def apply_dark_theme(fig, ax):
    ax.set_facecolor("#111317")
    fig.patch.set_facecolor("#111317")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#cfdf3d", labelsize=10)
    ax.yaxis.label.set_color("#cfdf3d")
    ax.xaxis.label.set_color("#cfdf3d")
    ax.title.set_color("#e7ebf3")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def stable_baseline(df_day: pd.DataFrame) -> float | None:
    """安定した基準値を選ぶ（10:00以降 / |val|>=EPS、なければ |val|>=EPS の最初、最後は最初の値）"""
    if df_day.empty:
        return None
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"])
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"])
    # 最終fallback：最初の値（条件なし）を返し、発散は calc_sane_pct 側で抑制
    return float(df_day.iloc[0]["val"])

def calc_sane_pct(base: float, close: float) -> float:
    """過剰な騰落率を防ぐ安全な計算"""
    try:
        denom = max(abs(base), abs(close), EPS)
        return (close - base) / denom * 100.0
    except Exception:
        return 0.0

def make_pct_series(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df
    if span == "1d":
        the_day = df["ts"].dt.floor("D").iloc[-1]
        df_day = df[df["ts"].dt.floor("D") == the_day].copy()
        if df_day.empty:
            return df_day
        base = stable_baseline(df_day)
        if base is None:
            return pd.DataFrame()
        df_day["pct"] = df_day["val"].apply(lambda x: calc_sane_pct(base, x))
        return df_day
    else:
        last = df["ts"].max()
        days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
        df_span = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
        if df_span.empty:
            return df_span
        base = float(df_span.iloc[0]["val"])
        df_span["pct"] = df_span["val"].apply(lambda x: calc_sane_pct(base, x))
        return df_span

def plot_one_span(df_pct: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(df_pct["ts"].values, df_pct["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    spans = ["1d", "7d", "1m", "1y"]
    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue
        df = load_csv(csv)
        df_pct = make_pct_series(df, span)
        if df_pct.empty or "pct" not in df_pct:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        title = f"{INDEX_KEY.upper()} ({span})"
        plot_one_span(df_pct, title, out_png)

if __name__ == "__main__":
    main()
