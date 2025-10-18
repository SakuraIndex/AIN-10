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

# ===== ダークテーマ（枠線なし、落ち着いた細グリッド） =====
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
    last = df["ts"].max()
    if span == "7d":
        return df[df["ts"] >= (last - pd.Timedelta(days=7))]
    if span == "1m":
        return df[df["ts"] >= (last - pd.Timedelta(days=30))]
    if span == "1y":
        return df[df["ts"] >= (last - pd.Timedelta(days=365))]
    return df


def pick_chart_baseline(df_span: pd.DataFrame, basis: str) -> tuple[float, str]:
    """
    チャート用の基準値を選ぶ。ain10_pct_post.py と同じ思想。
    """
    if df_span.empty:
        return 0.0, "n/a"

    if basis.startswith("stable@"):
        try:
            hhmm = basis.split("@", 1)[1]
            day = df_span["ts"].dt.floor("D").max()
            t0 = pd.Timestamp(day.strftime("%Y-%m-%d") + f" {hhmm}")
            d = df_span.copy()
            d["abs_diff"] = (d["ts"] - t0).abs()
            d = d[d["abs_diff"] <= pd.Timedelta(minutes=10)]
            if not d.empty:
                return float(d.sort_values("abs_diff").iloc[0]["val"]), basis
            return float(df_span.iloc[0]["val"]), f"{basis}(fallback=open)"
        except Exception:
            return float(df_span.iloc[0]["val"]), f"{basis}(fallback=open)"

    # open/prev_close/その他 → 先頭
    return float(df_span.iloc[0]["val"]), basis


def plot_one_span(df: pd.DataFrame, span: str, basis: str, out_png: Path):
    """
    レベル→基準との差分％に変換して描画
    change_% = (val - baseline) * 100
    """
    if df.empty:
        return

    base, basis_note = pick_chart_baseline(df, basis)
    y = (df["val"].astype(float) - base) * 100.0

    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark_theme(fig, ax)

    ax.set_title(f"{INDEX_KEY.upper()} ({span}) - {basis_note}", fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)

    ax.plot(df["ts"].values, y.values, linewidth=2.6, color="#ff615a")

    # X軸の日時ロケータ
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=50))

    # 過度な外れ値で縦軸が潰れるのを軽減（中央値±IQR×6 でソフトにクリップ）
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 6 * iqr, q3 + 6 * iqr
    if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    # チャートの基準（ワークフローから環境変数で渡してもOK）
    basis = os.environ.get("AIN10_CHART_BASIS", "stable@10:00")  # "open" / "prev_close" / "stable@HH:MM"

    for span in ["1d", "7d", "1m", "1y"]:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists():
            continue
        df = load_csv(csv)
        df_span = subset_for_span(df, span)
        if df_span.empty:
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_one_span(df_span, span, basis, out_png)


if __name__ == "__main__":
    import math
    main()
