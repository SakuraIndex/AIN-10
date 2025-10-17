#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIN-10: level + percent(range-based) charts & stats

- 1d/7d/1m/1y のPNGを生成（余計な注釈なし、タイトルのみ）
- post_intraday.txt: "Δ(level) と Δ%(range)" を併記。レンジ極小は N/A。
- *_stats.json: {"index_key","pct_1d","delta_level","scale","updated_at"}
  pct_1d はレンジ極小時は null

※ range-based % = (last - first) / (high - low) * 100
  level系インジケータが 0 付近を跨いでも暴れにくい定義。
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# ===== 設定 =====
INDEX_KEY = "ain10"  # ← AIN-10
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ---- 統一カラー（ダーク） ----
FIG_BG = "#0e0f13"  # figure背景
AX_BG  = "#0b0c10"  # 軸エリア背景
GRID   = "#2a2e3a"
LINE   = "#ff6b63"
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
    "font.size": 12,
})

EPS = 1e-9  # ゼロ割回避

# ===== 共通 =====
def _load_df() -> pd.DataFrame:
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _calc_delta_and_pct_range(df: pd.DataFrame, col: str):
    """Δ(level) と レンジ基準の％を返す。レンジ極小は pct=None。"""
    if df.empty:
        return None, None, None, None
    first = float(df[col].iloc[0])
    last  = float(df[col].iloc[-1])
    high  = float(df[col].max())
    low   = float(df[col].min())
    delta_level = last - first
    rng = high - low
    if rng <= EPS:
        pct = None  # レンジが実質ゼロ → ％は定義しない
    else:
        pct = (delta_level / rng) * 100.0
    return delta_level, pct, (first, last), (low, high)

# ===== 作図 =====
def _plot(df, col, out_png, title):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景塗りつぶしを確実に
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)

    # 枠・グリッド
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)

    # 軸 & タイトル（数値の注釈は載せない）
    ax.set_title(title, color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    # ライン
    ax.plot(df.index, df[col], linewidth=1.7, color=LINE)

    # 余白を少し確保して保存
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def gen_all():
    df = _load_df()
    if df.empty:
        return
    col = df.columns[-1]

    # 期間別
    _plot(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

# ===== 出力（テキスト/JSON） =====
def write_stats_and_post():
    df = _load_df()
    if df.empty:
        # 何も書かない
        return

    col = df.columns[-1]
    delta_level, pct, (first, last), (low, high) = _calc_delta_and_pct_range(df, col)

    # --- stats.json ---
    stats_payload = {
        "index_key": INDEX_KEY,
        "pct_1d": (None if pct is None else float(pct)),  # null or number
        "delta_level": (None if delta_level is None else float(delta_level)),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False), encoding="utf-8"
    )

    # --- post_intraday.txt ---
    # 表記例: "AIN10 1d: Δ=-0.856514 (level) Δ%=+12.34% (basis=range first-row valid=...)"
    # レンジ極小時は Δ%=N/A
    if pct is None:
        pct_str = "N/A"
    else:
        pct_str = f"{pct:+.2f}%"

    # 期間（first行のUTC→last行のUTC）
    start_ts = df.index[0].strftime("%Y-%m-%dT%H:%M:%S%z")
    end_ts   = df.index[-1].strftime("%Y-%m-%dT%H:%M:%S%z")
    # GitHub上では %z が +0000 形式になることがあるため、見た目整え
    start_ts = start_ts[:-2] + ":" + start_ts[-2:] if len(start_ts) >= 5 else start_ts
    end_ts   = end_ts[:-2]   + ":" + end_ts[-2:]   if len(end_ts) >= 5 else end_ts

    line = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level) "
        f"Δ%={pct_str} (basis=range first-row valid={start_ts}->{end_ts})"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line + "\n", encoding="utf-8")

# ===== エントリ =====
if __name__ == "__main__":
    gen_all()
    write_stats_and_post()
