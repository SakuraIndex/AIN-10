#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------- settings ----------
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV   = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV  = OUTDIR / f"{INDEX_KEY}_intraday.csv"
POST_INTRADAY = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
STATS_JSON    = OUTDIR / f"{INDEX_KEY}_stats.json"

# unified colors
FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"
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

# ---------- helpers ----------
def _load_df() -> pd.DataFrame:
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    # 数値化 & 完全NaN行の除去
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _latest_session(df: pd.DataFrame) -> pd.DataFrame:
    """同一日(ローカル日付)の最新セッションのみ抽出。"""
    last_date = df.index[-1].date()
    return df[df.index.date == last_date]

def _first_valid(s: pd.Series):
    return s.dropna().iloc[0] if s.dropna().size else None

def _last_valid(s: pd.Series):
    return s.dropna().iloc[-1] if s.dropna().size else None

def _fmt_signed(x: float, digits=6) -> str:
    # グラフの数値は視認性重視、小数点はデータ桁に合わせて調整
    return f"{x:+.{digits}f}"

def _fmt_pct(x: float, digits=2) -> str:
    return f"{x:+.{digits}f}%"

def _plot(df: pd.DataFrame, col: str, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    # 背景を確実に塗る
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    # 線のみ。余計なテキストは描かない
    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)

    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG)
    plt.close(fig)

# ---------- main tasks ----------
def gen_all():
    df = _load_df()
    col = df.columns[-1]  # 最新列を描画対象に
    # 範囲は適当に広めに確保 (indexは連続でなくてもOK)
    _plot(df.tail(1000), col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7*1000), col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df.tail(30*1000), col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def compute_intraday_change():
    df = _load_df()
    col = df.columns[-1]
    ses = _latest_session(df)[col]
    open_ = _first_valid(ses)
    close = _last_valid(ses)

    if open_ is None or close is None:
        return None, None, None, None

    delta_level = float(close - open_)
    # 基準は始値(open)。abs(open) でスケールして百分率
    pct_1d = float(100.0 * (close - open_) / (abs(open_) if abs(open_) > 1e-12 else 1e-12))

    # タイムスタンプ（表示用）
    start_ts = ses.dropna().index[0]
    end_ts   = ses.dropna().index[-1]
    return delta_level, pct_1d, start_ts, end_ts

def write_stats_and_post():
    delta_level, pct_1d, start_ts, end_ts = compute_intraday_change()
    nowz = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # stats.json
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_1d if pct_1d is not None else None,
        "delta_level": delta_level if delta_level is not None else None,
        "scale": "level",
        "basis": "open",  # ここがポイント: 始値基準
        "updated_at": nowz,
    }
    STATS_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # post_intraday.txt
    if pct_1d is not None:
        line = (
            f"{INDEX_KEY.upper()} 1d: Δ={_fmt_signed(delta_level)} (level) "
            f"A%={_fmt_pct(pct_1d)} "
            f"(basis=open first-row valid={start_ts.isoformat()}->{end_ts.isoformat()})"
        )
    else:
        line = f"{INDEX_KEY.upper()} 1d: Δ=N/A (level) A%=N/A (no valid session)"
    POST_INTRADAY.write_text(line + "\n", encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats_and_post()
