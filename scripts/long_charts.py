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
HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# --- 統一カラー ---
FIG_BG = "#0e0f13"  # figure 全体の背景
AX_BG  = "#0b0c10"  # 軸エリア背景
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"  # 視認性を少し上げる
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

EPS = 1e-6

def _robust_pct(first: float, last: float):
    """
    始値が小さすぎる場合は対称パーセンテージ変化に自動切替。
    戻り値: (pct, basis)
      pct: float (%)
      basis: "open" | "symmetric" | "n/a"
    """
    if first is None or last is None or (math.isnan(first) or math.isnan(last)):
        return None, "n/a"

    # どちらも極小: 実質変化なし扱い
    if abs(first) < EPS and abs(last) < EPS:
        return 0.0, "n/a"

    # 通常の%変化を試みる
    if abs(first) >= 0.2:  # しきい値は経験的（0付近での暴発を回避）
        pct = (last / first - 1.0) * 100.0
        return float(pct), "open"

    # 始値が小さすぎる → 対称%に切替
    denom = (abs(first) + abs(last)) / 2.0
    if denom < EPS:
        return 0.0, "n/a"
    pct = ((last - first) / denom) * 100.0
    return float(pct), "symmetric"

def _plot(df, col, out_png, title):
    if df.empty:
        # 空でも古い画像が残るよりは真っさらに差し替え
        fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
        fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(AX_BG)
        ax.set_title(f"{title}", color=FG)
        ax.text(0.5, 0.5, "no data", color=FG, transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
        fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # --- 背景 ---
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(AX_BG); ax.patch.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=AX_BG, zorder=-10))

    # --- スタイル ---
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    # --- プロット ---
    ax.plot(df.index, df[col], color=LINE, linewidth=1.7)

    # --- 保存 ---
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def _load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    # 時系列で昇順整列（安全）
    df = df.sort_index()
    return df

def _slice_same_day(df: pd.DataFrame):
    """最新行と同一日だけを抽出（intraday 1d用）。該当しなければ末尾1000件を返す。"""
    if df.empty: return df
    last_day = df.index[-1].date()
    day_df = df[df.index.date == last_day]
    if len(day_df) >= 2:
        return day_df
    return df.tail(1000)

def gen_all():
    df = _load_df()
    if df.empty:
        for k, t in [("1d","1d"),("7d","7d"),("1m","1m"),("1y","1y")]:
            _plot(df, None, OUTDIR / f"{INDEX_KEY}_{k}.png", f"{INDEX_KEY.upper()} ({t})")
        return

    col = df.columns[-1]

    # 1d: 当日スライス（intradayがない場合は末尾近傍）
    df_1d = _slice_same_day(df)
    _plot(df_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")

    # 7d/1m/1y は全体DFから範囲で（データ粒度に依存するため件数ベース）
    _plot(df.tail(7*1000),  col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df.tail(30*1000), col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df,              col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats():
    df = _load_df()
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,
        "delta_level": None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if df.empty:
        (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    col = df.columns[-1]
    day = _slice_same_day(df)
    if len(day) >= 2:
        first = float(day[col].iloc[0])
        last  = float(day[col].iloc[-1])
        delta = last - first
        pct, basis = _robust_pct(first, last)

        payload.update({
            "pct_1d": pct,
            "delta_level": delta,
            "basis": basis,
        })
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats()
