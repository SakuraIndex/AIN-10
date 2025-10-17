#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- dark theme
- dynamic line color
- level と % の併記（%は安全ガード付き）
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math

import pandas as pd
import numpy as np
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
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"  # ゼロ近傍・判定不能のとき

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
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return FLAT
    delta = s.iloc[-1] - s.iloc[0]
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=_trend_color(df[col]), linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex に、数値列へ強制変換して NA を除去。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _pick_index_column(df: pd.DataFrame) -> str:
    # 既存Ain-10は末尾列が本体想定
    return df.columns[-1]

# ------------------------
# % 計算（安全ガード）
# ------------------------
def _safe_pct_from_level(first: float, last: float, series: pd.Series) -> float | None:
    """
    (last/first - 1) * 100 を基本としつつ、以下をガード:
      - first が 0 またはニアゼロ
      - first に対して last が極端（スパイク）で、相対値が不自然
    不適切なら None を返す（表示は N/A に）。
    """
    if first is None or last is None:
        return None
    if not (math.isfinite(first) and math.isfinite(last)):
        return None

    # ニアゼロ判定: 絶対値が 1e-6 未満、または系列の中央値に対して 2% 未満
    med = float(np.nanmedian(np.abs(pd.to_numeric(series, errors="coerce").values)))
    near_zero_threshold = max(1e-6, 0.02 * med) if math.isfinite(med) and med > 0 else 1e-6
    if abs(first) < near_zero_threshold:
        return None

    pct = (last / first - 1.0) * 100.0

    # 極端値ガード（±100%を大幅に超える場合はデータ異常の可能性が高い）
    if abs(pct) > 100.0:
        return None
    return float(pct)

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    if df.empty:
        return
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)")

# ------------------------
# stats (level + pct) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    AIN-10 は level 系の指数。
    - delta_level: 最後の level - 最初の level（1d ウィンドウ）
    - pct_1d: (last/first - 1)*100（安全ガードで無理筋なら None）
    """
    df = _load_df()
    if df.empty:
        return
    col = _pick_index_column(df)

    # 1dウィンドウ（最大1000点）で first/last を取る
    w = df.tail(1000)[col].dropna()
    first_valid_ts = None
    last_valid_ts  = None

    if not w.empty:
        first_valid_ts = w.index[0].isoformat()
        last_valid_ts  = w.index[-1].isoformat()

    # level差分
    delta_level = None
    pct = None
    if len(w) >= 2:
        first = float(w.iloc[0])
        last  = float(w.iloc[-1])
        delta_level = last - first
        pct = _safe_pct_from_level(first, last, w)

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": None if delta_level is None else round(delta_level, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # human-friendly marker
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A\n", encoding="utf-8")
    else:
        pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_str} "
            f"(basis first-row valid={first_valid_ts}->{last_valid_ts})\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
