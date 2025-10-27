#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"

def main(csv, out_png, index_key="AIN-10", dt_col="Datetime"):
    raw = pd.read_csv(csv)
    # 日時をJSTインデックス化（緩め）
    if dt_col in raw.columns:
        dt = pd.to_datetime(raw[dt_col], utc=True, errors="coerce")
        dt = dt.dt.tz_convert(JST)
        raw.index = dt
        raw = raw.drop(columns=[dt_col])
    else:
        raw.index = pd.RangeIndex(len(raw))

    # 数値列平均
    cols = [c for c in raw.columns if pd.to_numeric(raw[c], errors="coerce").notna().mean() >= 0.8]
    s = raw[cols].mean(axis=1, skipna=True) if cols else pd.Series([0.0])

    # ％想定へ整形（0〜1の値が多ければ*100）
    q = float(np.quantile(np.abs(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy()), 0.95))
    if q < 0.5:
        s = s * 100.0

    s = pd.to_numeric(s, errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")
    ax.plot(s.index, s.values, linewidth=2.0)
    ax.set_title(f"{index_key} Intraday (fallback)", color="#DDDDDD")
    ax.set_xlabel("Time", color="#BBBBBB")
    ax.set_ylabel("Change vs Prev Close (%)", color="#BBBBBB")
    ax.tick_params(colors="#BBBBBB")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

if __name__ == "__main__":
    # usage: fallback_intraday.py CSV OUT_PNG [INDEX_KEY] [DT_COL]
    csv = sys.argv[1]
    out_png = sys.argv[2]
    index_key = sys.argv[3] if len(sys.argv) >= 4 else "AIN-10"
    dt_col = sys.argv[4] if len(sys.argv) >= 5 else "Datetime"
    main(csv, out_png, index_key, dt_col)
