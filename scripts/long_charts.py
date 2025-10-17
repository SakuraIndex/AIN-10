#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- ダークテーマ
- 1d/7d/1m/1y のPNGを出力
- 1日騰落「レベル差(Δ)」と「％(Δ%)」を併記して出力
- ゼロ割/異常値は安全に N/A フォールバック
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# constants / paths
# ----------------------------------
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ----------------------------------
# plotting style (dark)
# ----------------------------------
DARK_BG = "#0e0f13"  # whole figure
DARK_AX = "#0b0c10"  # axes patch
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply_dark(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(first: float | None, last: float | None) -> str:
    if first is None or last is None:
        return FLAT
    delta = last - first
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

def _save_line(df: pd.DataFrame, col: str, out_png: Path, title: str,
               first_for_color: float | None = None) -> None:
    fig, ax = plt.subplots()
    _apply_dark(ax, title)
    last_val = pd.to_numeric(df[col], errors="coerce").dropna()
    if last_val.empty:
        # 何も描けない場合でも空画像を保存(更新トリガにする)
        fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return
    color = _trend_color(first_for_color, last_val.iloc[-1])
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ----------------------------------
# data loading
# ----------------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    AIN-10の本体列を推定。候補が無ければ最後の列を使う。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "ain10", "ainindex", "ain10index", "ain10level", "ain_10", "ain"
    }
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば優先。先頭列を DatetimeIndex、数値列は to_numeric。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

# ----------------------------------
# intraday % logic (level → ratio)
# ----------------------------------
def _first_valid_for_ratio(s: pd.Series) -> float | None:
    """
    ％計算の分母に使う「最初の有効値」を選ぶ。
    きわめて小さい値(≈0)はゼロ割・暴騰表記を招くので除外。
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    for v in s:
        if abs(v) >= 1e-9:  # 閾値は必要に応じ調整
            return float(v)
    return None

def _last_valid(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])

def _calc_intraday_change(level_series: pd.Series) -> tuple[float | None, float | None, float | None]:
    """
    戻り値: (first, last, pct)
      - first/last: その日の最初/最後のレベル（有効）
      - pct: (last/first - 1) * 100（firstが極小/0なら None）
    """
    first = _first_valid_for_ratio(level_series)
    last  = _last_valid(level_series)
    if first is None or last is None:
        return first, last, None
    try:
        pct = (last / first - 1.0) * 100.0
    except ZeroDivisionError:
        pct = None
    return first, last, pct

# ----------------------------------
# charts + stats
# ----------------------------------
def gen_pngs_and_stats() -> dict:
    df = _load_df()
    col = _pick_index_column(df)

    # -- 1dウィンドウ: 直近(=intraday)を広めに確保
    tail_1d = df.tail(1000)
    # 方向色判定用に first/last を計算
    first, last, pct = _calc_intraday_change(tail_1d[col])

    # 画像
    _save_line(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png",
               f"{INDEX_KEY.upper()} (1d level)", first_for_color=first)
    _save_line(df.tail(7 * 1000), col, OUTDIR / f"{INDEX_KEY}_7d.png",
               f"{INDEX_KEY.upper()} (7d level)", first_for_color=None)
    _save_line(df, col, OUTDIR / f"{INDEX_KEY}_1m.png",
               f"{INDEX_KEY.upper()} (1m level)", first_for_color=None)
    _save_line(df, col, OUTDIR / f"{INDEX_KEY}_1y.png",
               f"{INDEX_KEY.upper()} (1y level)", first_for_color=None)

    # 統計JSON（サイト読み取り用）
    payload = {
        "index_key": INDEX_KEY,
        # pct_1d は ％。None の場合は JSON では null になる
        "pct_1d": None if pct is None else round(float(pct), 6),
        # 表示用の「レベル差」（便利値）
        "delta_level": None if (first is None or last is None) else round(float(last - first), 6),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "first": first, "last": last, "pct": pct
    }

def write_marker(first: float | None, last: float | None, pct: float | None) -> None:
    """
    投稿テキスト（人間向け）の整形: Δ(=レベル差) と Δ%(=％表記) を併記。
    """
    basis = "first-row valid"
    if first is not None and last is not None:
        delta = last - first
        delta_str = f"{delta:+.6f}"
    else:
        delta_str = "N/A"

    pct_str = "N/A" if pct is None else f"{pct:+.2f}%"

    # 時間範囲（可読補助）
    valid_range = ""
    try:
        df = _load_df()
        t0 = df.index.min().isoformat()
        t1 = df.index.max().isoformat()
        valid_range = f" (basis {basis} valid={t0}->{t1})"
    except Exception:
        pass

    text = f"{INDEX_KEY.upper()} 1d: Δ={delta_str} (level)  Δ%={pct_str}{valid_range}\n"
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(text, encoding="utf-8")

# ----------------------------------
# main
# ----------------------------------
if __name__ == "__main__":
    res = gen_pngs_and_stats()
    write_marker(res["first"], res["last"], res["pct"])
