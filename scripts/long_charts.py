#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- 1d/7d/1m/1y 画像を更新（dark theme, 動的ライン色）
- 1dのテキスト/JSONは「レベル差」と「％」を併記
- ％は基準値が小さい/符号が跨る等は N/A（異常な爆発値を防止）
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
FLAT    = "#9aa3af"  # 方向感なし

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

def _trend_color(first: float, last: float) -> str:
    if pd.isna(first) or pd.isna(last):
        return FLAT
    d = last - first
    if d > 0:
        return GREEN
    if d < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    if df.empty:
        return
    first = pd.to_numeric(df[col], errors="coerce").dropna()
    if first.empty:
        return
    first_val = first.iloc[0]
    last_val  = first.iloc[-1]

    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=_trend_color(first_val, last_val), linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    AIN-10 本体列を推定。既知が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "ain10", "ain10index", "ain10mean", "ain10level", "ain"
    }
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday を優先。先頭列を DatetimeIndex、
    数値列へ強制変換して NA 行は削除。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

# ------------------------
# % calculation guard
# ------------------------
def _safe_pct(first: float, last: float, min_abs: float = 0.5):
    """
    ％の定義: (last - first) / |first| * 100
    ただし以下は N/A （爆発/無意味な％を排除）
      - first が 0 に近い（|first| < min_abs）
      - 符号を跨ぐ（first*last <= 0）
      - NaN
    戻り値: (pct: float|None, reason: str|None)
    """
    if pd.isna(first) or pd.isna(last):
        return None, "no-data"
    if abs(first) < min_abs:
        return None, "small-baseline"
    if first * last <= 0:
        return None, "sign-change"
    pct = (last - first) / abs(first) * 100.0
    return float(pct), None

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    # いずれも「level」を描画（AIN-10 はレベル指数）
    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)")

# ------------------------
# stats (+ post)  level + %
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _first_last(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.iloc[0]), float(s.iloc[-1])

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    first, last = _first_last(df.tail(1000), col)  # 直近（≒1d）の範囲で評価
    delta = None if first is None or last is None else (last - first)

    pct, reason = _safe_pct(first, last) if delta is not None else (None, "no-data")

    payload = {
        "index_key": INDEX_KEY,
        # ％はガードにより None のことがある（サイト側で未表示扱い）
        "pct_1d": None if pct is None else round(pct, 6),
        # レベル差は常に出す
        "delta_level": None if delta is None else round(delta, 6),
        # AIN-10 は「level」スケール
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # human readable (post)
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    basis = "basis first-row valid"
    if delta is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A ({basis})\n", encoding="utf-8")
    else:
        delta_s = f"{delta:+.6f}"
        if pct is None:
            pct_s = "N/A"
        else:
            pct_s = f"{pct:+.2f}%"
        # df.index はすでに時系列なので先頭・末尾で期間も添える
        try:
            rng = df.tail(1000).index
            span = f"{rng[0].isoformat()}->{rng[-1].isoformat()}"
        except Exception:
            span = "n/a"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_s} (level)  Δ%={pct_s} ({basis}={span})\n",
            encoding="utf-8"
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
