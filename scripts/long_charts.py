#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCT_MODE_1D = True  # 1d のみ％表示に変換（7d/1m/1y は従来どおり level）

# ---- Dark style（枠線なし・落ち着いたグリッド）
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

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def subset_for_span(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if span == "1d":
        last_day = df["ts"].dt.floor("D").max()
        return df[df["ts"].dt.floor("D") == last_day]
    elif span == "7d":
        end = df["ts"].max()
        return df[df["ts"] >= (end - pd.Timedelta(days=7))]
    elif span == "1m":
        end = df["ts"].max()
        return df[df["ts"] >= (end - pd.Timedelta(days=30))]
    elif span == "1y":
        end = df["ts"].max()
        return df[df["ts"] >= (end - pd.Timedelta(days=365))]
    return df

def pick_10am_value(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    day = df["ts"].dt.floor("D").max()
    target = day + pd.Timedelta(hours=10)
    window = (df["ts"] >= target - pd.Timedelta(minutes=5)) & (df["ts"] <= target + pd.Timedelta(minutes=5))
    cand = df.loc[window]
    if cand.empty:
        return None
    i = (cand["ts"] - target).abs().idxmin()
    return float(cand.loc[i, "val"])

def transform_to_pct_1d(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """1d を % に変換（分母は stable@10:00 → fallback: median_abs@1d）"""
    basis = "n/a"
    if df.empty:
        return df.copy(), basis
    denom = pick_10am_value(df)
    if denom is not None and abs(denom) >= 1e-3:
        basis = "stable@10:00"
        base_ref = denom
    else:
        med_abs = float(df["val"].abs().median())
        if med_abs > 1e-6:
            basis = "median_abs@1d"
            base_ref = df.iloc[0]["val"]  # 実線の中心は初値近傍、分母は中央値
            denom = med_abs
        else:
            return df.copy(), "n/a"

    out = df.copy()
    out["val"] = (out["val"] - base_ref) / max(abs(denom), 1e-6) * 100.0
    return out, basis

def plot_one_span(df: pd.DataFrame, span: str, out_png: Path):
    title = f"{INDEX_KEY.upper()} ({span})"

    # 1d だけ％変換
    y_label = "Index (level)"
    if span == "1d" and PCT_MODE_1D:
        df, basis = transform_to_pct_1d(df)
        y_label = "Change (%)"
        title += "" if basis == "n/a" else f" – {basis}"

    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)

    ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    # x 軸：日付ロケータ/フォーマッタ
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    # y 軸：1d が％のときは % 表示
    if span == "1d" and PCT_MODE_1D:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    for span in ["1d", "7d", "1m", "1y"]:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue
        df_all = load_csv(csv)
        df = subset_for_span(df_all, span)
        if df.empty:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(df, span, out_png)

if __name__ == "__main__":
    main()
