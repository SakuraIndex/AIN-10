#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

# ========= 基本設定 =========
INDEX_KEY = "ain10"                 # ← AIN-10
SCALE     = "level"                 # 値のスケール（level / pct など）
OUTDIR    = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ========= スタイル =========
FIG_BG = "#0e0f13"   # 図全体の背景
AX_BG  = "#0b0c10"   # 軸エリア背景
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"   # 既存の赤トーンに合わせる
FG     = "#e7ecf1"

plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor":   AX_BG,
    "savefig.facecolor":FIG_BG,
    "savefig.edgecolor":FIG_BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "axes.titlecolor":  FG,
})

# ========= 共通ユーティリティ =========
def _load_df() -> pd.DataFrame:
    """
    intraday があればそれを、無ければ history を読み込む。
    先頭列を日時 index、残り列は数値に強制。
    """
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _latest_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    最後列を採用（列名とともに返す）
    """
    col = df.columns[-1]
    return df[col].dropna(), col

# ========= %算出（range 基準） =========
def compute_range_pct(s: pd.Series) -> dict:
    """
    当日（読み込んだ intraday 区間）の first/last と range を使って
    A%（= 100*(last-first)/range）を出す。
    """
    if len(s) == 0:
        return {"delta_level": None, "pct_1d": None, "basis": None,
                "first_ts": None, "last_ts": None}

    first_ts, last_ts = s.index[0], s.index[-1]
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    smin, smax = float(s.min()), float(s.max())
    rng = smax - smin

    delta = last - first
    if rng and abs(rng) > 0:
        pct = 100.0 * (delta / rng)
    else:
        pct = 0.0

    # 実務的に見栄えを保つため ±150% にソフトクリップ
    if pd.notna(pct):
        pct = max(min(pct, 150.0), -150.0)

    return {
        "delta_level": delta,
        "pct_1d": pct,
        "basis": "range",
        "first_ts": first_ts.isoformat(),
        "last_ts": last_ts.isoformat(),
    }

# ========= 描画 =========
def _plot(s: pd.Series, out_png: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景を明示的に
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=AX_BG, zorder=-10))

    # 軸まわり
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)

    # ラベル（余計な注記は出さない）
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    # 線
    ax.plot(s.index, s.values, linewidth=1.6, color=LINE)

    # 保存
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG,
                edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

def gen_all():
    df = _load_df()
    s, col = _latest_series(df)

    # 窓別に保存
    _plot(s.tail(1000), OUTDIR / f"{INDEX_KEY}_1d.png",  f"{INDEX_KEY.upper()} (1d)",  f"Index ({SCALE})")
    _plot(s.tail(7*1000), OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)",  f"Index ({SCALE})")
    _plot(s.tail(30*1000), OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", f"Index ({SCALE})")
    _plot(s, OUTDIR / f"{INDEX_KEY}_1y.png",               f"{INDEX_KEY.upper()} (1y)", f"Index ({SCALE})")

# ========= 出力（stats / テキスト） =========
def write_stats_and_post():
    df = _load_df()
    s, col = _latest_series(df)

    met = compute_range_pct(s)

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": met["pct_1d"],          # range 基準の%
        "delta_level": met["delta_level"],
        "scale": SCALE,
        "basis": met["basis"],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # ポスト用（1行）
    if met["pct_1d"] is None or pd.isna(met["pct_1d"]):
        pct_str = "N/A"
    else:
        pct_str = f"{met['pct_1d']:+.2f}%"

    first_ts = met["first_ts"]
    last_ts  = met["last_ts"]
    line = (f"{INDEX_KEY.upper()} 1d: Δ={met['delta_level']:+.6f} ({SCALE})  "
            f"A%={pct_str} (basis=range first-row valid={first_ts}->{last_ts})\n")
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line, encoding="utf-8")

if __name__ == "__main__":
    gen_all()
    write_stats_and_post()
