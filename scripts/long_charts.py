#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoDateLocator, AutoDateFormatter


INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# スパン定義
SPANS = ["1d", "7d", "1m", "1y"]


def _load_csv_generic(path: Path) -> pd.DataFrame:
    """先頭列=時刻, 2列目=値 として解釈して DataFrame[ts,val] を返す。"""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV columns insufficient: {path}")
    ts_col = df.columns[0]
    val_col = df.columns[1]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], errors="coerce"),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna(subset=["ts"])
    return out.sort_values("ts").reset_index(drop=True)


def _read_span_csv(span: str) -> pd.DataFrame:
    return _load_csv_generic(OUT_DIR / f"{INDEX_KEY}_{span}.csv")


def _apply_dark_style(ax: plt.Axes):
    """落ち着いた黒ベース / 目にうるさくないグリッド / 枠線なし。"""
    # 背景
    ax.set_facecolor("#0E1116")               # 深いダークグレー
    ax.figure.set_facecolor("#0E1116")

    # 枠線(スパイン)除去
    for spine in ax.spines.values():
        spine.set_visible(False)

    # グリッドは控えめに
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.22, which="both")

    # 目盛り色を淡色に
    ax.tick_params(colors="#C9CCD5", labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # 軸ラベル・タイトル色
    ax.set_xlabel("Time", color="#DADDE5", labelpad=8)
    ax.set_ylabel("Index (level)", color="#DADDE5", labelpad=8)
    ax.title.set_color("#FFFFFF")


def _line_color():
    # 落ち着いた赤 (#ff6b6b系は明る過ぎるので少し暗め)
    return "#FF6B6B"


def plot_span(span: str):
    df = _read_span_csv(span)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    _apply_dark_style(ax)

    # 日付軸
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    ax.plot(df["ts"], df["val"], linewidth=2.25, color=_line_color())

    ax.set_title(f"{INDEX_KEY.upper()} ({span})", fontsize=20, pad=14)
    out = OUT_DIR / f"{INDEX_KEY}_{span}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for span in SPANS:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if csv.exists():
            plot_span(span)


if __name__ == "__main__":
    main()
