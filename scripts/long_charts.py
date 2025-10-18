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

EPS = 0.2  # 小さすぎる分母回避

# ---- ダークテーマ ----
def apply_dark_theme(fig, ax):
    ax.set_facecolor("#111317")
    fig.patch.set_facecolor("#111317")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#cfd3dc", labelsize=10)
    ax.yaxis.label.set_color("#cfd3dc")
    ax.xaxis.label.set_color("#cfd3dc")
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

def choose_baseline_for_day(df_day: pd.DataFrame) -> tuple[float | None, str]:
    # 1) open
    if not df_day.empty:
        open_val = float(df_day.iloc[0]["val"])
        if abs(open_val) >= EPS:
            return open_val, "open"
    # 2) stable@10:00
    mask = (df_day["ts"].dt.hour > 10) | ((df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0))
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"]), "stable@10:00"
    # 3) それでもダメなら最初の |val|>=EPS
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"]), "first|val|>=EPS"
    return None, "no_pct_col"

def make_pct_series(df: pd.DataFrame, span: str) -> tuple[pd.DataFrame, str]:
    """span のデータを取り出し、%系列に変換して返す"""
    if df.empty:
        return df.copy(), "no_pct_col"

    # span 抽出
    if span == "1d":
        the_day = df["ts"].dt.floor("D").iloc[-1]
        df_span = df[df["ts"].dt.floor("D") == the_day].copy()
    elif span == "7d":
        last = df["ts"].max()
        df_span = df[df["ts"] >= (last - pd.Timedelta(days=7))].copy()
    elif span == "1m":
        last = df["ts"].max()
        df_span = df[df["ts"] >= (last - pd.Timedelta(days=30))].copy()
    elif span == "1y":
        last = df["ts"].max()
        df_span = df[df["ts"] >= (last - pd.Timedelta(days=365))].copy()
    else:
        df_span = df.copy()

    if df_span.empty:
        return df_span, "no_pct_col"

    # 1d のときだけ当日ベースで % 化、それ以外は “区間先頭” を基準に%化
    if span == "1d":
        base, note = choose_baseline_for_day(df_span)
    else:
        base = float(df_span.iloc[0]["val"])
        note = "span_start"
        if abs(base) < EPS:
            # span でもゼロ対策
            cand = df_span.loc[df_span["val"].abs() >= EPS]
            if cand.empty:
                return df_span, "no_pct_col"
            base = float(cand.iloc[0]["val"])
            note = "first|val|>=EPS"

    if base is None or abs(base) < EPS:
        return df_span, "no_pct_col"

    df_span["pct"] = (df_span["val"] - base) / abs(base) * 100.0
    return df_span, note

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
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    # y 目盛りは読みやすく
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
        df_pct, note = make_pct_series(df, span)
        if df_pct.empty or "pct" not in df_pct:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        title = f"{INDEX_KEY.upper()} ({span})" + ("" if span != "1d" else f"  -  {note}")
        plot_one_span(df_pct, title, out_png)

if __name__ == "__main__":
    main()
