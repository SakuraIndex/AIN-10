#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats (final fix)
- ダークテーマ
- 1d/7d/1m/1y PNG確実生成
- Δ(level) + Δ%(percent) 両方出力
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- 基本設定 ---
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# --- カラー設定 ---
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply(ax, title: str):
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT
    delta = s.iloc[-1] - s.iloc[0]
    return GREEN if delta > 0 else RED if delta < 0 else FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str):
    fig, ax = plt.subplots()
    _apply(ax, title)
    if df.empty:
        ax.plot([0], [0], color=FLAT, linewidth=1.0)  # 空でも描画
    else:
        ax.plot(df.index, df[col], color=_trend_color(df[col]), linewidth=1.6)
    fig.canvas.draw()
    fig.savefig(
        out_png,
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
        transparent=False,
    )
    print(f"[chart] saved: {out_png}")
    plt.close("all")

def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("CSV not found for AIN-10")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def gen_pngs():
    df = _load_df()
    col = df.columns[-1]
    _save(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats_and_marker():
    df = _load_df()
    col = df.columns[-1]
    today = df.tail(1000)[col].dropna()
    if today.empty:
        print("[warn] no data for stats")
        return

    first, last = today.iloc[0], today.iloc[-1]
    delta = last - first

    EPS = 0.05  # 0近傍を緩和
    pct = (last / first - 1) * 100 if abs(first) >= EPS else None

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "delta_level": round(delta, 6),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    first_ts, last_ts = df.index.min(), df.index.max()
    pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
    txt = f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level)  Δ%={pct_str} (basis {first_ts}->{last_ts})"
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(txt, encoding="utf-8")
    print(f"[stats] written: Δ={delta:+.6f}, Δ%={pct_str}")

if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
