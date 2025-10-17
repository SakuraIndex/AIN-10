#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json
import math

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"
POST_INTRADAY_TXT = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
STATS_JSON = OUTDIR / f"{INDEX_KEY}_stats.json"

# --- dark theme ---
FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"
FG     = "#e7ecf1"

plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "savefig.facecolor": FIG_BG,
})

# しきい値
EPS = 1e-4  # これ未満は0扱い

def safe_pct(first: float, last: float):
    """符号跨ぎ・小基準のときは None (N/A)"""
    delta = last - first
    if abs(first) < EPS or abs(last) < EPS or first * last <= 0:
        return None
    pct = (delta / abs(first)) * 100.0
    if abs(pct) > 1000:
        return None
    return pct

def plot_chart(df, col, out_png, title):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG, fontsize=13)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)
    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)

def load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def calc_stats(df, col):
    f = float(df[col].iloc[0])
    l = float(df[col].iloc[-1])
    delta = l - f
    pct = safe_pct(f, l)
    return f, l, delta, pct

def generate_all():
    df = load_df()
    col = df.columns[-1]

    # 各期間出力
    plot_chart(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    plot_chart(df.tail(7 * 1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    plot_chart(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    plot_chart(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats_and_post():
    df = load_df()
    col = df.columns[-1]
    _, _, delta, pct = calc_stats(df.tail(1000), col)

    pct_val = "N/A" if pct is None else f"{pct:+.2f}%"
    # JSON
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": round(delta, 6),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    }
    STATS_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 投稿TXT
    t0 = df.index[0].isoformat()
    t1 = df.index[-1].isoformat()
    msg = f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level)  A%={pct_val} (basis {t0}->{t1})"
    POST_INTRADAY_TXT.write_text(msg, encoding="utf-8")

if __name__ == "__main__":
    generate_all()
    write_stats_and_post()
