#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- dark theme
- 1d は level と % を併記（安定化ロジック付き）
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply(ax, title: str) -> None:
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

def _trend_color(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return FLAT
    return GREEN if s.iloc[-1] - s.iloc[0] > 0 else RED

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=_trend_color(df[col]), linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    """
    intraday 優先、無ければ history。数値へ強制変換して NA 行は落とす。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("AIN-10: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _pick_index_column(df: pd.DataFrame) -> str:
    # AIN-10 は基本最後の列が本体
    return df.columns[-1]

# ------------------------
# % の安定化ロジック
# ------------------------
def _open_like_for_pct(s: pd.Series, floor: float = 0.2) -> float | None:
    """
    “オープン近似” を選ぶ：
      1) 先頭が十分大きいなら先頭
      2) 先頭から30本以内で |x|>=floor の最初の値
      3) 全体の |x|中央値
      -> 最終的に |基準| が floor 未満なら None
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    first = float(s.iloc[0])
    if abs(first) >= floor:
        return first
    head = s.iloc[:30]
    nz = head[head.abs() >= floor]
    if not nz.empty:
        return float(nz.iloc[0])
    med = float(s.abs().median())
    if med >= floor:
        # 符号は先頭値に合わせる（符号無しでも分母には問題ないが一応）
        return med if first >= 0 else -med
    return None

def _delta_level_and_pct_1d(s: pd.Series) -> tuple[float | None, float | None]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None, None
    first = float(s.iloc[0])
    last  = float(s.iloc[-1])
    delta = last - first

    base = _open_like_for_pct(s)
    if base is None:
        pct = None
    else:
        # 分母が極端に小さい時の暴騰暴落誤判定を回避
        if abs(base) < 1e-6:
            pct = None
        else:
            pct = (last - base) / base * 100.0
    return delta, pct

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

# ------------------------
# stats + post marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    delta_level, pct_1d = _delta_level_and_pct_1d(df[col])

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None else round(float(pct_1d), 6),
        "delta_level": None if delta_level is None else round(float(delta_level), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # human-readable marker
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A\n", encoding="utf-8")
    else:
        if pct_1d is None:
            marker.write_text(
                f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%=N/A "
                f"(basis first-row valid={df.index[0]}->{df.index[-1]})\n",
                encoding="utf-8",
            )
        else:
            marker.write_text(
                f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_1d:+.2f}% "
                f"(basis first-row valid={df.index[0]}->{df.index[-1]})\n",
                encoding="utf-8",
            )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
