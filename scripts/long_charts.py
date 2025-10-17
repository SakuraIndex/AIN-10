#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10 charts + stats
- ダークテーマ（全指数で統一）
- 1d/7d/1m/1y の「level」チャート
- 1d の変化量を level と %（= level差×100）で併記
- ライン色：上げ=GREEN / 下げ=RED / 判定不能=FLAT
- NaN/欠損に強い基準選定（最初の有効値を基準にする）
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
FLAT    = "#9aa3af"  # ゼロ近傍・判定不能のとき

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_AX,
    "savefig.facecolor": DARK_BG,
    "savefig.edgecolor": DARK_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG_TEXT,
    "xtick.color": FG_TEXT,
    "ytick.color": FG_TEXT,
})

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

def _trend_color(delta_level: float | None) -> str:
    if delta_level is None:
        return FLAT
    if delta_level > 0:
        return GREEN
    if delta_level < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str, delta_for_color: float | None) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    color = _trend_color(delta_for_color)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex に、数値列へ強制変換して NA を除去。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError(f"{INDEX_KEY}: neither intraday nor history csv found.")
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _pick_index_column(df: pd.DataFrame) -> str:
    """
    AIN-10 の本体列を推定。既知候補が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {"ain10", "ainindex", "ain10index", "sakuraain10"}
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

# ------------------------
# summarization (1d delta, etc.)
# ------------------------
def _first_valid(s: pd.Series) -> tuple[float | None, pd.Timestamp | None]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return None, None
    return float(s2.iloc[0]), s2.index[0]

def _last_valid(s: pd.Series) -> tuple[float | None, pd.Timestamp | None]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return None, None
    return float(s2.iloc[-1]), s2.index[-1]

def _intraday_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    1d 用のウィンドウを取得。
    intraday.csv がある前提では「当日（UTC）」の行が並ぶ想定。
    データ側の粒度がバラけても安全に、直近 ~1000 行で代替。
    """
    return df.tail(1000)

def _summarize(df: pd.DataFrame, col: str) -> dict:
    """
    1d の変化量:
      delta_level = last - first_valid
      delta_pct   = delta_level * 100  （％ポイント換算）
    ※ division は行わない（ゼロ割回避）
    """
    win = _intraday_window(df)
    first_val, first_ts = _first_valid(win[col])
    last_val,  last_ts  = _last_valid(win[col])

    delta_level = None
    delta_pct   = None
    if first_val is not None and last_val is not None:
        delta_level = last_val - first_val
        delta_pct   = delta_level * 100.0

    return {
        "win": win,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "delta_level": delta_level,
        "delta_pct": delta_pct,
    }

def _iso_z(dt: pd.Timestamp | None) -> str:
    if dt is None:
        return "N/A"
    # pandas.Timestamp → ISO (Z)
    return dt.tz_convert("UTC").isoformat(timespec="minutes").replace("+00:00", "Z") \
        if dt.tzinfo is not None else dt.tz_localize("UTC").isoformat(timespec="minutes").replace("+00:00", "Z")

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # color 判定に 1d 変化を使う
    s = _summarize(df, col)
    delta_for_color = s["delta_level"]

    # 1d
    _save(s["win"], col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", delta_for_color)

    # 7d/1m/1y は純変化で色判定（上げ → 緑、下げ → 赤）
    def _delta_window(wdf: pd.DataFrame) -> float | None:
        fv, _ = _first_valid(wdf[col]); lv, _ = _last_valid(wdf[col])
        return None if fv is None or lv is None else (lv - fv)

    win_7d = df.tail(7 * 1000)
    _save(win_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)", _delta_window(win_7d))
    _save(df,     col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)", _delta_window(df))
    _save(df,     col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)", _delta_window(df))

# ------------------------
# stats + marker (level + %)
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    s = _summarize(df, col)

    payload = {
        "index_key": INDEX_KEY,
        # % は「レベル差×100」。サイト側は％表示を期待。
        "pct_1d": None if s["delta_pct"] is None else round(float(s["delta_pct"]), 6),
        # 併記用に level 差も保持（任意で参照できるように）
        "delta_level": None if s["delta_level"] is None else round(float(s["delta_level"]), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # 投稿用マーカー（人間可読）
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if s["delta_level"] is None:
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A (basis first-row invalid)\n",
            encoding="utf-8",
        )
    else:
        basis = f"{_iso_z(s['first_ts'])}->{_iso_z(s['last_ts'])}"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={s['delta_level']:+.6f} (level)  Δ%={s['delta_pct']:+.2f}% (basis first-row valid={basis})\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
