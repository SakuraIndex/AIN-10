#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN10 charts + stats (ASTRA4準拠, レベル系列, dark theme)
内部では%は出力しない。X投稿用の%は別スクリプト(post_for_x.py)で算出。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 定数 / パス設定
# ============================
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ============================
# ダークテーマ設定
# ============================
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_AX,
    "savefig.facecolor": DARK_BG,
    "axes.edgecolor": GRID,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "xtick.color": FG_TEXT,
    "ytick.color": FG_TEXT,
    "axes.labelcolor": FG_TEXT,
    "axes.titlecolor": FG_TEXT,
})

# ============================
# 共通ユーティリティ
# ============================
def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True)
    ax.tick_params(colors=FG_TEXT)
    ax.set_xlabel("Time", color=FG_TEXT)
    ax.set_ylabel("Index Level", color=FG_TEXT)
    ax.set_title(title, fontsize=12, color=FG_TEXT)

def _trend_color(series: pd.Series) -> str:
    """
    開始値と終値の比較で線色を決定。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT
    if s.iloc[-1] > s.iloc[0]:
        return GREEN
    elif s.iloc[-1] < s.iloc[0]:
        return RED
    return FLAT

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

# ============================
# データ読み込み
# ============================
def _load_df() -> pd.DataFrame:
    """
    intraday があれば優先、なければ history を使う。
    indexをDatetimeIndex化し、数値列へ変換。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN10: CSV not found")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

# ============================
# チャート生成
# ============================
def _plot(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    color = _trend_color(df[col])
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)

def gen_all_charts() -> None:
    df = _load_df()
    col = df.columns[-1]

    # intradayの最新データを別ファイルとして保存（post_for_x用）
    df.tail(1000).to_csv(OUTDIR / f"{INDEX_KEY}_1d.csv")

    # チャート出力
    _plot(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7 * 1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

# ============================
# 統計ファイル（ASTRA4仕様）
# ============================
def write_stats() -> None:
    df = _load_df()
    col = df.columns[-1]
    last = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1] if len(df) else None

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,            # ASTRA4準拠: 常に None
        "delta_level": float(last) if last is not None else None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": _now_utc_iso(),
    }

    # JSON出力
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # テキストマーカー
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    marker.write_text(f"{INDEX_KEY.upper()} 1d: A%=N/A (basis n/a)\n", encoding="utf-8")

# ============================
# main
# ============================
if __name__ == "__main__":
    gen_all_charts()
    write_stats()
