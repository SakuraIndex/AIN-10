#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats  (pct scale, unified dark theme)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ------------------------
# unified dark theme (strict)
# ------------------------
DARK_BG  = "#0e0f13"   # figure 背景
DARK_AX  = "#0b0c10"   # 軸エリア背景
FG_TEXT  = "#e7ecf1"
GRID     = "#2a2e3a"
LINE_COL = "#00c2a8"   # AIN-10 既存のライン色

# ここで matplotlib の既定を強制固定（ズレ防止）
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_AX,
    "savefig.facecolor": DARK_BG,
    "savefig.edgecolor": DARK_BG,
    "savefig.transparent": False,  # 透明PNGを禁止
})

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)

    # 念のため個別にも徹底的に指定
    fig.patch.set_facecolor(DARK_BG)
    fig.patch.set_alpha(1.0)
    ax.set_facecolor(DARK_AX)
    ax.patch.set_alpha(1.0)

    for sp in ax.spines.values():
        sp.set_color(GRID)

    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index / Value", color=FG_TEXT, fontsize=10)

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=LINE_COL, linewidth=1.6)
    # 透明禁止 & 背景色を明示（念押し）
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=DARK_BG,
        edgecolor=DARK_BG,
        transparent=False,
    )
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    df = df.dropna(how="all")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = df.columns[-1]
    _save(df.tail(1000),  col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df,             col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df,             col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

# ------------------------
# stats (pct)
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    AIN-10 は intraday が %（百分率）を直接保持（例: 0.95 は +0.95%）
    """
    df = _load_df()
    col = df.columns[-1]

    pct = None
    if len(df.index) > 0:
        last = df[col].iloc[-1]
        if pd.notna(last):
            pct = float(last)

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
