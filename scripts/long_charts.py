#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- level 指数：Δlevel と Δ% を併記
- Δ% = (last - first) / |first| * 100   ※ |first| が極小なら N/A
- ダークテーマ／トレンド色は R-BANK9/ASTRA4 準拠
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
      - mode="intraday": 末尾値の符号で (+)緑 / (-)赤 / 0はFLAT
      - mode="window"  : 期間の純変化 (last - first) で判定
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT

    if mode == "intraday":
        last = s.iloc[-1]
        if last > 0:
            return GREEN
        if last < 0:
            return RED
        return FLAT

    first = s.iloc[0]
    last  = s.iloc[-1]
    delta = last - first
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
    """
    AIN-10 本体列を推定。既知候補が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "ain10", "ain10index", "ain10level", "ain", "ainindex"
    }
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex に、数値化して NA を除去。
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

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)      # intraday 相当（同日が入る想定）
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)", mode="intraday")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", mode="window")

# ------------------------
# stats (level + pct) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _delta_level_and_pct(df: pd.DataFrame, col: str):
    """
    当日（=読み込んだ intraday 窓）の first/last から
    Δlevel と Δ% を算出。|first| が極小なら pct は None。
    """
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty or len(s) < 2:
        return None, None

    first = float(s.iloc[0])
    last  = float(s.iloc[-1])
    delta = last - first

    eps = 1e-6
    if abs(first) < eps:
        pct = None
    else:
        pct = (delta / abs(first)) * 100.0
    return delta, pct

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # intraday の範囲（実際に読み込んだ窓）で first/last を取る
    tail_1d = df.tail(1000)
    start_ts = tail_1d.index[0] if len(tail_1d) else None
    end_ts   = tail_1d.index[-1] if len(tail_1d) else None

    delta, pct = _delta_level_and_pct(tail_1d, col)

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": None if delta is None else round(delta, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta is None:
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ=N/A (level) Δ%=N/A\n",
            encoding="utf-8",
        )
    else:
        pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
        basis = ""
        if start_ts is not None and end_ts is not None:
            basis = f" (basis first-row valid={start_ts.isoformat()}->{end_ts.isoformat()})"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level) Δ%={pct_str}{basis}\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
