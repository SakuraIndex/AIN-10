#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIN-10: 1d/7d/1m/1y のチャート生成 + 統計出力（%は異常値回避のハイブリッド方式）
- %は以下のハイブリッドで安定化
  1) 通常:    A% = (last - first) / abs(first) * 100 （基準=first-row）
  2) 例外時:  |first| が小さすぎる場合は range 法に自動切替
              A% = (last - first) / (max - min) * 100 （基準=range）
  → どちらを使ったかは出力テキストに basis=... で明記
- 画像上の注記（Δや%のテキスト）は描かない（タイトルのみ）
- *_stats.json には level 差分と % を併記
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# 統一カラー
FIG_BG = "#0e0f13"  # figure 背景
AX_BG  = "#0b0c10"  # 軸エリア背景
GRID   = "#2a2e3a"
LINE   = "#ff6b6b"  # AIN10 は視認性重視で暖色に
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
    "grid.color": GRID,
})

# =========================
# ユーティリティ
# =========================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _load_df() -> pd.DataFrame:
    """
    intraday があればそれを優先、無ければ history を使う。
    """
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _compute_changes(s: pd.Series):
    """
    Δ(level) と A%(ハイブリッド) を計算し、basis を返す。
    - 通常は first-row 基準
    - first が小さ過ぎる場合は range 法に自動切替
    """
    s = s.dropna()
    if len(s) < 2:
        return 0.0, 0.0, "n/a"

    first = float(s.iloc[0])
    last  = float(s.iloc[-1])
    delta_level = last - first

    smax = float(s.max())
    smin = float(s.min())
    rng  = smax - smin

    # first が十分大きいかを “range 比” で判定（しきい値は 10%）
    eps = max(1e-12, 0.10 * (rng if rng > 0 else 1.0))

    if abs(first) >= eps:
        pct = (delta_level / abs(first)) * 100.0
        basis = "first-row"
    else:
        # range 法（rng==0 の場合は 0%）
        pct = (delta_level / rng * 100.0) if rng > 0 else 0.0
        basis = "range"

    return delta_level, pct, basis

def _plot(df: pd.DataFrame, col: str, out_png: Path, title: str):
    """
    タイトル以外の注記は描かない。背景はダーク、線は見やすい赤系。
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景を念のため明示
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)

    # グリッド & 軸スタイル
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title, color=FG, pad=12)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)

    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)

    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=FIG_BG,
        edgecolor=FIG_BG,
        transparent=False,
    )
    plt.close(fig)

# =========================
# メイン処理
# =========================
def gen_all():
    df = _load_df()
    col = df.columns[-1]  # 最新カラムを描画対象に

    # 1d/7d/1m/1y
    _plot(df.tail(1000),     col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(df.tail(7*1000),   col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(df,                col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(df,                col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

def write_stats_and_post():
    df = _load_df()
    col = df.columns[-1]
    s   = df[col].dropna()

    if len(s) == 0:
        # 何も無ければ空で返す
        payload = {
            "index_key": INDEX_KEY,
            "pct_1d": None,
            "delta_level": None,
            "scale": "level",
            "updated_at": _now_utc_iso(),
        }
        (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return

    # 直近 1d 相当（ファイルの粒度次第だが、ここでは全体を「当日窓」として扱う）
    delta_level, pct_1d, basis = _compute_changes(s)

    # JSON（ダッシュボード/機械読取用）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_1d,                 # %（ハイブリッド・basisはテキスト出力で明示）
        "delta_level": delta_level,       # level差分
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # テキスト（人間向けログ）
    # 例: "AIN10 1d: Δ=-0.856514 (level)  A%=-34.56% (basis=range first-row valid=...->...)"
    first_ts = s.index[0].isoformat()
    last_ts  = s.index[-1].isoformat()
    line = (
        f"{INDEX_KEY.upper()} 1d: "
        f"Δ={delta_level:+.6f} (level)  "
        f"A%={pct_1d:+.2f}% (basis={basis} "
        f"first-row valid={first_ts}->{last_ts})"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line, encoding="utf-8")

def main():
    gen_all()
    write_stats_and_post()

if __name__ == "__main__":
    main()
