#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats  (dark theme unified)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ---- unified colors (他指数と合わせる) ----
DARK_BG = "#0e0f13"   # 図(figure)全体の背景
DARK_AX = "#0b0c10"   # 軸(axes)の背景
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
LINE    = "#00c2a8"   # 既存どおり(陽/陰はサイト全体方針に従い将来変更可)

# 保存時に “必ず塗り潰し・透過しない” をデフォルト化
plt.rcParams["savefig.facecolor"] = DARK_BG
plt.rcParams["savefig.edgecolor"] = DARK_BG

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)

    # 背景を明示
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
    ax.plot(df.index, df[col], color=LINE, linewidth=1.6)

    # ここがポイント：transparent=False かつ face/edge を DARK_BG 指定
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=DARK_BG,
        edgecolor=DARK_BG,
        transparent=False,
    )
    plt.close(fig)

def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("no csv")
    # 数値列に統一
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def gen_pngs() -> None:
    df = _load_df(); col = df.columns[-1]
    _save(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats() -> None:
    """
    AIN-10 は intraday が %単位 → 騰落率[%] = last_value
    """
    df = _load_df(); col = df.columns[-1]
    pct = float(df[col].iloc[-1]) if len(df) else None
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else pct,
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(
        f"{INDEX_KEY.upper()} 1d: {'N/A' if pct is None else f'{pct:+.2f}%'}\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    gen_pngs()
    write_stats()
