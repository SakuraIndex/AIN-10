#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黒ベースの落ち着いたチャート生成（枠線なし / grid控えめ）
- 入力: docs/outputs/<index>_{1d,7d,1m,1y}.csv
- 出力: docs/outputs/<index>_{1d,7d,1m,1y}.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()
OUT_DIR = "docs/outputs"

# ---- theme (dark) ----
BG   = "#0d0e11"
FG   = "#e0e0e0"
GRID = "#2d2f33"
LINE = "#ff5c5c"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": BG,            # 外枠の白線を消す
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.25,
    "font.size": 11,
})

def _detect_time_col(cols):
    # 代表的な候補を広めに拾う
    keys = ["datetime", "timestamp", "time", "date", "ts"]
    for c in cols:
        cl = c.strip().lower()
        if any(k in cl for k in keys):
            return c
    return cols[0]

def _detect_value_col(cols, time_col, index_key):
    # 時刻列以外から候補選択
    prefer = [
        index_key, index_key.replace("_", "-"), index_key.upper(), index_key.replace("_", "-").upper(),
        "ain10", "ain-10", "value", "index", "score", "close", "price", "y"
    ]
    norm = [c.strip() for c in cols]
    # 第一候補: 明示候補にマッチ
    for c in cols:
        cc = c.strip()
        if cc != time_col.strip() and (cc in prefer or cc.lower() in [p.lower() for p in prefer]):
            return c
    # 第二候補: 時刻列以外の最後
    fallback = [c for c in cols if c.strip() != time_col.strip()]
    return fallback[-1] if fallback else cols[-1]

def load_series(csv_path, index_key):
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["ts","val"])
    # 列名の余白除去
    df.columns = [c.strip() for c in df.columns]

    tcol = _detect_time_col(df.columns)
    vcol = _detect_value_col(df.columns, tcol, index_key)

    # リネームに依存せず、常に新規列として作る（KeyError: 'ts' 回避）
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    val = pd.to_numeric(df[vcol], errors="coerce")

    out = pd.DataFrame({"ts": ts, "val": val}).dropna(subset=["ts","val"]).sort_values("ts")
    return out

def plot_chart(csv_path, out_path, label):
    if not os.path.exists(csv_path):
        return
    df = load_series(csv_path, INDEX_KEY)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12,5), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.plot(df["ts"], df["val"], color=LINE, lw=1.8)
    ax.set_title(f"AIN10 ({label})", color=FG, fontsize=17, weight="bold", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=8))
    # 時間が粗密どちらでも見やすい2段表示
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%Y-%m-%d"))

    # 念のため枠を不可視に
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for span in ["1d","7d","1m","1y"]:
        csv_p = f"{OUT_DIR}/{INDEX_KEY}_{span}.csv"
        png_p = f"{OUT_DIR}/{INDEX_KEY}_{span}.png"
        plot_chart(csv_p, png_p, span)

if __name__ == "__main__":
    main()
