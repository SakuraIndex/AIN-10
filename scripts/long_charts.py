#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- level基準のチャート
- 騰落率は当日 first_valid→last の相対変化。ただし起点が小さ過ぎる時は N/A
- ダークテーマ統一 & 自動線色
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
FLAT    = "#9aa3af"  # ほぼ横ばい/判定不能

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    # 目障りにならない薄いグリッド（白線っぽくならない色）
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series, mode: str) -> str:
    """線色の判定:
       - mode="intraday": 最後の値の符号 (+)緑 / (-)赤 / 0はFLAT
       - mode="window"  : 期間の純変化 (last-first) で判定
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
    # window
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
    """AIN-10の本体列を推定（既知候補優先、なければ最後の列）"""
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {"ain10", "index", "level", "ainindex", "ain10level"}
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """intraday があれば優先。先頭列を DatetimeIndex に、数値化してNA除去。"""
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

    # 1d は intraday として最後の符号で色判定
    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", mode="intraday")

    # 7d/1m/1y は純変化で色判定
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)", mode="window")

# ------------------------
# stats (level + optional pct) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _today_slice(df: pd.DataFrame) -> pd.DataFrame:
    """UTC基準で同日のデータを抽出（最初と最後の有効値を得るため）"""
    if df.empty:
        return df
    last_ts = df.index.max()
    day_start = last_ts.normalize()
    return df.loc[(df.index >= day_start) & (df.index <= last_ts)]

def write_stats_and_marker() -> None:
    df = _load_df()
    if df.empty:
        # 空なら安全に終了
        payload = {
            "index_key": INDEX_KEY,
            "pct_1d": None,
            "delta_level": None,
            "scale": "level",
            "updated_at": _now_utc_iso(),
        }
        (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level) Δ%=N/A\n", encoding="utf-8")
        return

    col = _pick_index_column(df)
    today = _today_slice(df[[col]]).dropna()

    delta_level = None
    pct = None
    first_ts = last_ts = None

    if not today.empty:
        first_val = float(today[col].iloc[0])
        last_val  = float(today[col].iloc[-1])
        first_ts  = today.index[0]
        last_ts   = today.index[-1]
        delta_level = last_val - first_val

        # ゼロ近傍での暴走回避（%は意味が薄いのでN/A）
        EPS = 0.5  # 起点が |first| < EPS のときは N/A（適宜調整）
        if abs(first_val) >= EPS:
            pct = (last_val / first_val - 1.0) * 100.0
        else:
            pct = None

    # stats.json
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

    # post_intraday marker（level と % を併記。%はN/A対応）
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A\n", encoding="utf-8")
    else:
        pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
        basis = ""
        if first_ts and last_ts:
            basis = f" (basis first-row valid={first_ts.isoformat()}->{last_ts.isoformat()})"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_str}{basis}\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
