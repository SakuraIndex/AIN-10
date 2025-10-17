#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 用：レベル指標（非価格）。%は表示しない。
- 1d/7d/1m/1y チャートをダークテーマで出力
- stats.json は Δ(level) のみ、pct_1d=None
- post_intraday.txt は Δ%=N/A と明記
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INDEX_KEY = "ain10"  # ← AIN-10 固定
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# --- ダークテーマ色 ---
FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6d6d"   # 見やすい赤
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

def _load_df() -> pd.DataFrame:
    """intraday があれば優先、なければ history。"""
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _plot(df: pd.DataFrame, value_col: str, out_png: Path, title: str):
    """余計な文字は付けず、タイトルのみ。毎回上書き保存。"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景の塗りを強制
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=AX_BG, zorder=-10))

    # 枠・メモリ
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_title(title, color=FG, pad=8)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    # 線
    ax.plot(df.index, df[value_col], linewidth=1.7, color=LINE)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG,
                edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def gen_all():
    """1d/7d/1m/1y を常に再生成。"""
    df = _load_df()
    col = df.columns[-1]  # 最新列を採用

    _plot(df.tail(1000),    col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7*1000),  col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df.tail(30*1000), col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df,               col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats_and_post():
    """
    AIN-10 は ‘level’ 指標。
    - Δ(level) は当日内の first→last の単純差
    - pct_1d は常に None（N/A）
    - basis は 'n/a' を明記
    """
    df = _load_df()
    col = df.columns[-1]
    last = float(df[col].iloc[-1]) if len(df) else None

    # “当日”範囲の first/last（intraday があればその範囲）
    if INTRADAY_CSV.exists() and len(df) > 0:
        first = float(df[col].iloc[0])
    else:
        # history しかない場合は直近2点で代替
        first = float(df[col].iloc[0]) if len(df) else None

    delta_level = None
    if first is not None and last is not None:
        delta_level = last - first

    # stats.json
    stats = {
        "index_key": INDEX_KEY,
        "pct_1d": None,               # ← % は出さない
        "delta_level": delta_level,
        "scale": "level",
        "basis": "n/a",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )

    # 投稿テキスト（intraday）
    # 例: "AIN10 1d: Δ=-0.8565 (level)  Δ%=N/A (basis n/a valid=start->end)"
    valid_from = df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else "n/a"
    valid_to   = df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else "n/a"
    line = (f"{INDEX_KEY.upper()} 1d: Δ={delta_level:.6f} (level)  "
            f"Δ%=N/A (basis n/a valid={valid_from}->{valid_to})")
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line + "\n", encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats_and_post()
