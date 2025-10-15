#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# --- 統一カラー ---
FIG_BG = "#0e0f13"  # figure 全体の背景
AX_BG  = "#0b0c10"  # 軸エリア背景
GRID   = "#2a2e3a"
LINE   = "#00c2a8"
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
})

def _plot(df, col, out_png, title):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # --- 背景を確実に塗る ---
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)

    # 背景矩形を明示的に描画（確実に塗る）
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        facecolor=AX_BG, zorder=-10
    ))

    # --- スタイル ---
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index / Value", color=FG)

    # --- プロット ---
    ax.plot(df.index, df[col], color=LINE, linewidth=1.6)

    # --- 保存 ---
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=FIG_BG,
        edgecolor=FIG_BG,
        transparent=False,
    )
    plt.close(fig)

def _load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def gen_all():
    df = _load_df(); col = df.columns[-1]
    _plot(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats():
    df = _load_df(); col = df.columns[-1]
    val = float(df[col].iloc[-1]) if len(df) else None
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": val,
        "scale": "pct",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats()
