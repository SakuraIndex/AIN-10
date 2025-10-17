#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# ==== 設定 ====
INDEX_KEY = "ain10"  # 出力名は "ain10_*.png" で統一（"ain_10" などにしない）
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# 色・見た目（統一）
FIG_BG = "#0e0f13"
AX_BG  = "#0b0c10"
GRID   = "#2a2e3a"
LINE   = "#ff6868"  # AIN系は見やすい赤に
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

# ==== ユーティリティ ====

def _load_df() -> pd.DataFrame:
    """intraday があれば優先。最後の列を数値化して返す。"""
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _symmetric_pct(a: float, b: float) -> float:
    """
    対称％変化：  200 * (b - a) / (|a| + |b|)
    - ゼロ跨ぎ・負の基準でも安定
    - 返り値は [-200, +200] に収まる
    """
    denom = abs(a) + abs(b)
    if denom == 0:
        return float("nan")
    return 200.0 * (b - a) / denom

def _last_window(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """直近 n 行（タイムスタンプ順）"""
    return df.tail(n)

def _calc_change(df: pd.DataFrame, col: str):
    """先頭→末尾のレベル差分と対称％を返す"""
    first = float(df[col].iloc[0])
    last  = float(df[col].iloc[-1])
    delta_level = last - first
    pct = _symmetric_pct(first, last)
    return first, last, delta_level, pct

def _plot(df: pd.DataFrame, col: str, out_png: Path, title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    # 背景
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.patch.set_facecolor(AX_BG)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=AX_BG, zorder=-10))
    # スタイル
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=FG)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), color=FG)
    ax.set_xlabel("Time", color=FG)
    ax.set_ylabel("Index (level)", color=FG)
    # 線
    ax.plot(df.index, df[col], color=LINE, linewidth=1.8)
    # 保存
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG, transparent=False)
    plt.close(fig)

# ==== 生成 ====

def gen_all():
    df = _load_df()
    if df.empty:
        return
    col = df.columns[-1]

    # 直近ウィンドウ（行数ベースで簡易に）
    df_1d = _last_window(df, 1000)    # 実運用は時間幅で切るならここを調整
    df_7d = _last_window(df, 7*1000)
    df_1m = df.copy()
    df_1y = df.copy()

    # 1d の注記（レベル差分 & 対称％）
    _, _, dlev_1d, dpct_1d = _calc_change(df_1d, col)
    sub_1d = f"Δ(level)={dlev_1d:+.6f}  Δ%(sym)={dpct_1d:+.2f}%"

    # 7d/1m/1y もタイトル下に注記を載せる
    _, _, dlev_7d, dpct_7d = _calc_change(df_7d, col)
    _, _, dlev_1m, dpct_1m = _calc_change(df_1m, col)
    _, _, dlev_1y, dpct_1y = _calc_change(df_1y, col)

    _plot(df_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", sub_1d)
    _plot(df_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)",
          f"Δ={dlev_7d:+.6f} / {dpct_7d:+.2f}%")
    _plot(df_1m, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)",
          f"Δ={dlev_1m:+.6f} / {dpct_1m:+.2f}%")
    _plot(df_1y, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)",
          f"Δ={dlev_1y:+.6f} / {dpct_1y:+.2f}%")

def write_stats_and_posts():
    df = _load_df()
    if df.empty:
        return
    col = df.columns[-1]

    # 1d窓で統計値
    df_1d = _last_window(df, 1000)
    first, last, delta_level, pct_sym = _calc_change(df_1d, col)

    # JSON（レベル差分と対称％の両方を保存）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": pct_sym,             # 対称％（-200〜+200）
        "delta_level": delta_level,    # レベル差分
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # 投稿用テキスト（レベル+％併記）
    # basis（期間）は先頭/末尾の時刻を埋め込む
    t0 = df_1d.index[0]
    t1 = df_1d.index[-1]
    line = (
        f"{INDEX_KEY.upper()} 1d:  Δ={delta_level:+.6f} (level)  Δ%={pct_sym:+.2f}%  "
        f"(basis first-row valid={t0.isoformat()}->{t1.isoformat()})"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line, encoding="utf-8")

def main():
    gen_all()
    write_stats_and_posts()

if __name__ == "__main__":
    main()
