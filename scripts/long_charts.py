#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

# ========= 基本設定（AIN-10 専用） =========
INDEX_KEY = "ain10"
SCALE = "level"              # レベル指標（%は出さない）
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ========= 統一ダークテーマ =========
FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"  # 視認性高い赤
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
    "axes.titlecolor": FG,
})

# ========= 共通ユーティリティ =========
def _load_df() -> pd.DataFrame:
    """intraday があれば優先。なければ history。"""
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _plot(df: pd.DataFrame, col: str, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景を確実に塗る
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=AX_BG, zorder=-10))

    # スタイル
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # ライン
    ax.plot(df.index, df[col], linewidth=1.8, color=LINE)

    # 余計なテキストオーバーレイは出さない（邪魔だった注記は削除）
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def gen_all():
    df = _load_df()
    if df.empty:
        return
    col = df.columns[-1]
    # 1d/7d/1m/1y（サンプル数に応じてトリム）
    _plot(df.tail(1000),        col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7 * 1000),    col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df.tail(30 * 1000),   col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df,                   col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats_and_posts():
    """
    - レベル差（Δlevel）のみを算出
    - %は未定義のため出さない（pct_1d = null）
    - ポストテキストも A%=N/A と明示
    """
    df = _load_df()
    col = df.columns[-1] if not df.empty else None
    last_val = float(df[col].iloc[-1]) if col and len(df) else None
    first_val = float(df[col].iloc[0]) if col and len(df) else None
    delta_level = float(last_val - first_val) if last_val is not None and first_val is not None else None

    # JSON（stats）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,                # ← %は未定義
        "delta_level": delta_level,    # レベル差のみ
        "scale": SCALE,                # "level"
        "basis": "n/a",                # 分母を持つ%計算はなし
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # intradayポスト（テキスト）
    # 例: "AIN10 1d: Δ=-0.123456 (level)  A%=N/A (basis n/a  valid=...->...)"
    started = df.index[0].strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "N/A"
    ended   = df.index[-1].strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "N/A"
    line = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  "
        f"A%=N/A (basis n/a valid={started}->{ended})"
        if delta_level is not None else
        f"{INDEX_KEY.upper()} 1d: A%=N/A (no data)"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line + "\n", encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats_and_posts()
