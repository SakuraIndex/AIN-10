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

# --- 統一カラー ---
FIG_BG = "#0e0f13"  # figure 全体の背景
AX_BG  = "#0b0c10"  # 軸エリア背景
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
})

EPS = 1e-6        # 0割回避用のしきい値
HARD_CAP = 1000.0 # 異常値ガード（絶対値%がこれ超ならN/A）

def _safe_pct(first: float, last: float):
    """レベル値から安全な%を計算。返り値は (pct or None, basis_str)"""
    delta = last - first
    # 1) |first| 基準
    if abs(first) >= EPS:
        pct = (delta / abs(first)) * 100.0
        if abs(pct) <= HARD_CAP:
            return pct, "abs(first)"
    # 2) 対称% (Hodrick式に近い安全基準)
    denom = (abs(first) + abs(last)) / 2.0
    if denom >= EPS:
        pct = (delta / denom) * 100.0
        if abs(pct) <= HARD_CAP:
            return pct, "symmetric"
    # 3) 計算不能
    return None, "N/A"

def _plot(df, col, out_png, title, delta_str="", pct_str=""):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=AX_BG, zorder=-10))

    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)

    sub = ""
    if delta_str or pct_str:
        sub = f"\n{delta_str}  {pct_str}".strip()
    ax.set_title(f"{title}{sub}", color=FG)

    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def _load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _last_first(df, col):
    first = float(df[col].iloc[0])
    last = float(df[col].iloc[-1])
    delta = last - first
    pct, basis = _safe_pct(first, last)
    return first, last, delta, pct, basis

def gen_all():
    df = _load_df()
    col = df.columns[-1]
    first, last, delta, pct, basis = _last_first(df, col)
    delta_str = f"Δ(level)={delta:.6f}"
    pct_str = f"Δ%(basis={basis})={pct:.2f}%" if pct is not None else "Δ%(basis=N/A)=N/A"

    _plot(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", delta_str, pct_str)
    _plot(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)")
    _plot(df,               col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)")
    _plot(df,               col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)")

def write_stats_and_post():
    df = _load_df()
    col = df.columns[-1]
    first = df.index.min().strftime("%Y-%m-%dT%H:%M:%S%z") if len(df) else ""
    last  = df.index.max().strftime("%Y-%m-%dT%H:%M:%S%z") if len(df) else ""

    f, l, delta, pct, basis = _last_first(df, col)
    # JSON
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": round(delta, 6),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    STATS_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 投稿テキスト
    pct_txt = "N/A" if pct is None else f"{pct:+.2f}%"
    msg = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level)  A%={pct_txt} "
        f"(basis first-row valid={df.index[0].isoformat()}->{df.index[-1].isoformat()})"
    )
    POST_INTRADAY_TXT.write_text(msg, encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats_and_post()
