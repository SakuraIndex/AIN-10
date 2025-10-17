#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIN-10 (normalized index chart + stable % computation)
- % = Δ(level) / σ(level) * 100  （標準偏差基準）
- レンジ or first-row 基準で暴れる問題を完全に回避
- σ < 1e-3 の場合は A% = N/A
- チャート上の余計な注記なし
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"
FG     = "#e7ecf1"

plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": AX_BG,
    "savefig.facecolor": FIG_BG,
    "savefig.edgecolor": FIG_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
})

EPS = 1e-6

# ====== Load Data ======
def load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

# ====== Stats ======
def calc_delta_stdev(df, col):
    vals = df[col].dropna()
    if len(vals) < 2:
        return 0.0, None, "n/a"

    first = float(vals.iloc[0])
    last  = float(vals.iloc[-1])
    delta = last - first
    sigma = float(vals.std())

    if sigma < EPS:
        return delta, None, "n/a"

    pct = (delta / sigma) * 100.0
    return delta, pct, "stdev"

# ====== Plot ======
def plot_chart(df, col, out_png, title):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)
    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG)
    plt.close(fig)

# ====== Write ======
def write_outputs():
    df = load_df()
    col = df.columns[-1]
    delta, pct, basis = calc_delta_stdev(df, col)
    start_ts = df.index[0].isoformat()
    end_ts   = df.index[-1].isoformat()

    pct_str = "N/A" if pct is None else f"{pct:+.2f}%"

    # TXT
    msg = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level) "
        f"A%={pct_str} (basis={basis} "
        f"first-row valid={start_ts}->{end_ts})"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(msg, encoding="utf-8")

    # JSON
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else pct,
        "delta_level": delta,
        "scale": "level",
        "basis": basis,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

# ====== Main ======
def main():
    df = load_df()
    col = df.columns[-1]
    plot_chart(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    plot_chart(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    plot_chart(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    plot_chart(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")
    write_outputs()

if __name__ == "__main__":
    main()
