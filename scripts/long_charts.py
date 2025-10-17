#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- dark theme
- dynamic line color (up=green / down=red / flat=gray)
- stats: delta_level + pct_1d(=(last/first-1)*100) with zero-guard
"""
from pathlib import Path
from datetime import datetime, timezone
import json
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
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

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

def _trend_color(series: pd.Series, mode: str) -> str:
    """
    線色の判定:
      - mode="intraday": 期間内の純変化 (last - first) で判定
      - mode="window"  : 同上（7d/1m/1y でも同じ）
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT
    delta = float(s.iloc[-1] - s.iloc[0])
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str, mode: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    color = _trend_color(df[col], mode=mode)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """AIN-10 本体列を推定。候補が無ければ最後の列。"""
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {"ain10", "ainindex", "ain10index"}
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """intraday 優先、無ければ history。先頭列を DatetimeIndex にして数値化。"""
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

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", mode="intraday")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)", mode="window")

# ------------------------
# stats (level Δ & pct) + post marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _first_last_non_na(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.iloc[0]), float(s.iloc[-1])

def write_stats_and_marker() -> None:
    """
    出力仕様:
      - delta_level = last - first  （水準差）
      - pct_1d      = (last / first - 1) * 100   （first==0 は N/A）
      - scale       = "level" で統一（グラフの縦軸は level）
      - post_intraday.txt は Δ( level ) と Δ%( pct ) を併記
    """
    df = _load_df()
    col = _pick_index_column(df)

    first, last = _first_last_non_na(df[col])
    delta_level = None
    pct = None
    basis = "first-row invalid"
    if first is not None and last is not None:
        delta_level = last - first
        if first != 0:
            pct = (last / first - 1.0) * 100.0
        basis = f"first-row valid={df.index[0].isoformat()}->{df.index[-1].isoformat()}"

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(float(pct), 6),
        "delta_level": None if delta_level is None else round(float(delta_level), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # marker: level + %
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A (basis {basis})\n", encoding="utf-8")
    else:
        pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_str} (basis {basis})\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
