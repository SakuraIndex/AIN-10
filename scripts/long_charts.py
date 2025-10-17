#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("docs/outputs")
OUT.mkdir(parents=True, exist_ok=True)
INDEX_KEY = "ain10"

def _load(name):
    p = OUT / f"{INDEX_KEY}_{name}.csv"
    return pd.read_csv(p)

def _numeric_two_cols(df):
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[ts_col, val_col]].rename(columns={ts_col:"ts", val_col:"value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])

def _plot(df, title, png_name):
    # ダーク & 赤線 & 枠線なし
    fig, ax = plt.subplots(figsize=(12,6), dpi=140)
    fig.patch.set_facecolor("#0c0f12")
    ax.set_facecolor("#0c0f12")
    ax.plot(df["ts"], df["value"], linewidth=2.2)  # 色はデフォ（後で赤）
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="-")
    ax.set_title(title, color="white", fontsize=20, pad=14)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Index (level)", color="white")
    ax.tick_params(colors="white")
    # 線色を赤に（切替システムがあってもここで固定）
    ax.lines[0].set_color("#ff6b6b")

    fig.tight_layout()
    fig.savefig(OUT / png_name, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_csv(name, df):
    df.to_csv(OUT / f"{INDEX_KEY}_{name}.csv", index=False)

def main():
    # 既に生成済みの 1d/7d/1m/1y CSV がある前提（このリポのパイプライン仕様）
    for span in ["1d", "7d", "1m", "1y"]:
        df = _numeric_two_cols(_load(span))
        _plot(df, f"{INDEX_KEY.upper()} ({span})", f"{INDEX_KEY}_{span}.png")

if __name__ == "__main__":
    main()
