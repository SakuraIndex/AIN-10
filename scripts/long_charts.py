#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats (pct scale, dark theme, auto color)
背景統一版（他指数と完全一致）
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ------------------------
# unified dark theme colors
# ------------------------
DARK_BG = "#0e0f13"   # 背景
DARK_AX = "#0b0c10"   # 軸エリア
FG_TEXT = "#e7ecf1"   # 文字
GRID    = "#2a2e3a"

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
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

    # --- ここで既存の色ロジックを保持 ---
    # 値の増減に応じて色を動的に決定（陽線→緑 / 陰線→赤）
    first, last = df[col].iloc[0], df[col].iloc[-1]
    color = "#00c873" if last >= first else "#ff6b6b"

    ax.plot(df.index, df[col], color=color, linewidth=1.6)

    # 背景を確実に統一
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)

def _pick_index_column(df: pd.DataFrame) -> str:
    def norm(s: str) -> str:
        return s.strip().lower().replace("-", "").replace("_", "")
    targets = {INDEX_KEY, "ain10index"}
    for c in df.columns:
        if norm(c) in targets:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    _save(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    pct = float(df[col].iloc[-1]) if len(df.index)>0 and pd.notna(df[col].iloc[-1]) else None

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
    marker.write_text(
        f"{INDEX_KEY.upper()} 1d: {'N/A' if pct is None else f'{pct:+.2f}%'}\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
