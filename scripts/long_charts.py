#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長期チャート生成スクリプト（完全安定版）

✅ 修正点：
- matplotlib バージョン差異対応（AutoDateLocator, AutoDateFormatter の import 安全化）
- チャートは黒背景・外枠なし・赤線・落ち着いたグリッド
- CSV カラム名に依存しない（先頭列=時刻, 2列目=値 として解釈）
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPANS = ["1d", "7d", "1m", "1y"]

# ====== デザイン設定 ======
BG = "#0d0e11"
FG = "#d7d7d7"
GRID = "#2c2f33"
LINE = "#ff5c5c"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": BG,  # 外枠線を削除
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.25,
    "font.size": 11,
})


# ====== CSV 読み込み ======
def load_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV列が足りません: {csv_path}")

    ts_col = df.columns[0]
    val_col = df.columns[1]

    df = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], errors="coerce"),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna(subset=["ts", "val"])
    return df.sort_values("ts").reset_index(drop=True)


# ====== チャート描画 ======
def plot_chart(csv_path: Path, out_path: Path, label: str):
    if not csv_path.exists():
        print(f"[WARN] Missing {csv_path}")
        return

    df = load_series(csv_path)
    if df.empty:
        print(f"[WARN] Empty data: {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(df["ts"], df["val"], color=LINE, lw=2.0)
    ax.set_title(f"{INDEX_KEY.upper()} ({label})", color=FG, fontsize=17, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # === 枠線削除 ===
    for spine in ax.spines.values():
        spine.set_visible(False)

    # === グリッド ===
    ax.grid(True, alpha=0.25)

    # === 軸フォーマット（日付軸） ===
    try:
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    except Exception:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%Y-%m-%d"))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for span in SPANS:
        csv_file = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        if csv_file.exists():
            plot_chart(csv_file, out_png, span)


if __name__ == "__main__":
    main()
