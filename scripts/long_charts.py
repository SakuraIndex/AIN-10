#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Dark style (枠線なし・落ち着いたグリッド) ----
def apply_dark_theme(fig, ax):
    # 背景
    ax.set_facecolor("#111317")         # 濃いグレー（黒に近い）
    fig.patch.set_facecolor("#111317")

    # 枠線を消す
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 目盛りの色/フォント
    ax.tick_params(colors="#cfd3dc", labelsize=10)
    ax.yaxis.label.set_color("#cfd3dc")
    ax.xaxis.label.set_color("#cfd3dc")
    ax.title.set_color("#e7ebf3")

    # グリッドは薄く
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    CSV の 1列目を時刻、2列目を値として読み込む。
    例: "Datetime,AIN_10"
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")

    # 列名を正規化
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    # パース
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def plot_one_span(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    # 軸
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Index (level)", labelpad=10)

    # 線（赤）
    ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    # x軸の日時ロケータ
    locator = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(AutoDateFormatter(locator))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50, prune=None))

    fig.tight_layout()
    # 重要：保存時に facecolor を維持、枠線を作らない
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def subset_for_span(df: pd.DataFrame, span: str) -> pd.DataFrame:
    """span: '1d'|'7d'|'1m'|'1y' の最新区間を抽出"""
    if span == "1d":
        last_day = df["ts"].dt.floor("D").max()
        return df[df["ts"].dt.floor("D") == last_day]
    elif span == "7d":
        last = df["ts"].max()
        return df[df["ts"] >= (last - pd.Timedelta(days=7))]
    elif span == "1m":
        last = df["ts"].max()
        return df[df["ts"] >= (last - pd.Timedelta(days=30))]
    elif span == "1y":
        last = df["ts"].max()
        return df[df["ts"] >= (last - pd.Timedelta(days=365))]
    else:
        return df

def main():
    spans = ["1d", "7d", "1m", "1y"]
    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            # CSV が無ければスキップ（1d だけ先にあってもOK）
            continue

        df = load_csv(csv)
        df_span = subset_for_span(df, span)
        if df_span.empty:
            continue

        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(df_span, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    main()
