#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# ── settings ──────────────────────────────────────────────────────────────
INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %基準の検出パラメータ（1dのみ使用）
PCT_MIN_ABS_BASE = 0.10        # 基準の最小絶対値
PCT_SEARCH_FROM  = "09:35"     # この時刻以降で最初の十分大きい値を基準にする

# ── theme ─────────────────────────────────────────────────────────────────
def apply_dark_theme(fig, ax):
    """黒ベース・枠線なし・落ち着いたグリッド"""
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

# ── io ────────────────────────────────────────────────────────────────────
def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs >=2 columns: {csv_path}")

    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ── helpers ──────────────────────────────────────────────────────────────
def subset_for_span(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df
    if span == "1d":
        last_day = df["ts"].dt.floor("D").max()
        return df[df["ts"].dt.floor("D") == last_day]
    last = df["ts"].max()
    if span == "7d":
        return df[df["ts"] >= (last - pd.Timedelta(days=7))]
    if span == "1m":
        return df[df["ts"] >= (last - pd.Timedelta(days=30))]
    if span == "1y":
        return df[df["ts"] >= (last - pd.Timedelta(days=365))]
    return df

def to_percent_series_1d(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """
    1dの値を%, 基準は:
      - 当日 09:35 以降で最初の |val| >= PCT_MIN_ABS_BASE
      - なければ先頭値が十分大きければそれを使用
      - どちらも満たさなければ None を返して level 表示にフォールバック
    """
    if df.empty:
        return None, "n/a"

    day = df["ts"].dt.floor("D").max()
    start = pd.Timestamp(f"{day.date()} {PCT_SEARCH_FROM}")
    cand = df[df["ts"] >= start]
    cand = cand[cand["val"].abs() >= PCT_MIN_ABS_BASE]

    if not cand.empty:
        base = float(cand.iloc[0]["val"])
        basis_note = f"first_nonzero@{PCT_SEARCH_FROM}"
    else:
        base = float(df.iloc[0]["val"])
        if abs(base) < PCT_MIN_ABS_BASE:
            return None, "no_pct_col"
        basis_note = "open"

    pct = (df["val"].astype(float) - base) / abs(base) * 100.0
    return pct, basis_note

# ── plotting ─────────────────────────────────────────────────────────────
def plot_one_span(df: pd.DataFrame, title: str, out_png: Path, span: str):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    if span == "1d":
        y, basis = to_percent_series_1d(df)
        if y is not None:
            ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
            ax.set_xlabel("Time", labelpad=10)
            ax.set_ylabel("Change (%)", labelpad=10)
            ax.plot(df["ts"].values, y.values, linewidth=2.6, color="#ff615a")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            # 基準が取れない日は level 表示でフォールバック
            ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
            ax.set_xlabel("Time", labelpad=10)
            ax.set_ylabel("Index (level)", labelpad=10)
            ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")
    else:
        ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
        ax.set_xlabel("Time", labelpad=10)
        ax.set_ylabel("Index (level)", labelpad=10)
        ax.plot(df["ts"].values, df["val"].values, linewidth=2.6, color="#ff615a")

    # X軸：日付ロケータ/フォーマッタ
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)

# ── main ─────────────────────────────────────────────────────────────────
def main():
    spans = ["1d", "7d", "1m", "1y"]
    print("[MARKER] long_charts.py started")

    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue

        df = load_csv(csv)
        df_span = subset_for_span(df, span)
        rows = len(df_span)
        if rows == 0:
            continue

        ts_min = df_span["ts"].min()
        ts_max = df_span["ts"].max()

        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        print(f"[MARKER] plotting span={span} rows={rows} ts_min={ts_min} ts_max={ts_max} -> {out_png}")
        plot_one_span(df_span, f"{INDEX_KEY.upper()} ({span})", out_png, span)
        print(f"[MARKER] saved figure: {out_png}")

    print("=== MARKER: long_charts finished ===")

if __name__ == "__main__":
    main()
