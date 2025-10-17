#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Long-term charts generator for index levels (1d/7d/1m/1y).

- Charts are LEVEL (not percent). Percent text is NOT drawn on the figure.
- Daily percent for X posting is computed separately and written to
  docs/outputs/<index>_post_intraday.txt and also saved in <index>_stats.json (pct_1d).
- Dark theme, red line, tidy axes.
"""

import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ------------ configuration -------------
OUTDIR = "docs/outputs"
INDEX_KEY = os.environ.get("INDEX_KEY", "ain10").lower()

# Input CSVs (already produced by your other jobs)
CSV_1D = os.path.join(OUTDIR, f"{INDEX_KEY}_1d.csv")
CSV_7D = os.path.join(OUTDIR, f"{INDEX_KEY}_7d.csv")
CSV_1M = os.path.join(OUTDIR, f"{INDEX_KEY}_1m.csv")
CSV_1Y = os.path.join(OUTDIR, f"{INDEX_KEY}_1y.csv")

CSV_INTRADAY = os.path.join(OUTDIR, f"{INDEX_KEY}_intraday.csv")  # for daily % posting

# Output images
PNG_1D = os.path.join(OUTDIR, f"{INDEX_KEY}_1d.png")
PNG_7D = os.path.join(OUTDIR, f"{INDEX_KEY}_7d.png")
PNG_1M = os.path.join(OUTDIR, f"{INDEX_KEY}_1m.png")
PNG_1Y = os.path.join(OUTDIR, f"{INDEX_KEY}_1y.png")

# Other outputs
STATS_JSON = os.path.join(OUTDIR, f"{INDEX_KEY}_stats.json")
POST_INTRADAY = os.path.join(OUTDIR, f"{INDEX_KEY}_post_intraday.txt")

# ------------ style (dark) -------------
def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#0b0f19",
        "savefig.facecolor": "#0b0f19",
        "axes.edgecolor": "#8892a6",
        "axes.labelcolor": "#c7d0e0",
        "xtick.color": "#c7d0e0",
        "ytick.color": "#c7d0e0",
        "grid.color": "#3b4252",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "font.size": 12,
    })

LINE_COLOR = "#ff5b5b"
# ---------------------------------------


def _is_time_col(name: str) -> bool:
    if name is None:
        return False
    n = name.lower()
    return n in ("ts", "time", "timestamp", "date", "datetime") or "time" in n or "date" in n


def load_series(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and return DataFrame with columns: ['ts', 'value'].
    Time column is auto-detected. Value column = the first non-time column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty CSV: {csv_path}")

    # detect time column
    time_col = None
    for c in df.columns:
        if _is_time_col(c):
            time_col = c
            break
    if time_col is None:
        # fallback: first column is time
        time_col = df.columns[0]

    # detect value column (first non-time)
    value_col = None
    for c in df.columns:
        if c != time_col:
            value_col = c
            break
    if value_col is None:
        raise ValueError(f"no value column in {csv_path}")

    # parse
    s = df[[time_col, value_col]].copy()
    s.columns = ["ts", "value"]
    # best-effort datetime parse
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce", utc=False)
    s = s.dropna(subset=["ts", "value"])
    s = s.sort_values("ts")
    s = s.reset_index(drop=True)
    return s


def render_level_chart(df: pd.DataFrame, title: str, outfile: str):
    """
    Draw simple line chart of 'value' vs 'ts' with dark theme and red line.
    Title only. No overlay stats on the figure.
    """
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["ts"], df["value"], LINE_COLOR, linewidth=2)
    ax.set_title(title, color="#e5ecf6", pad=12, fontsize=16)

    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # tidy spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # nice margins
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def try_render(csv_path: str, title: str, out_png: str):
    try:
        df = load_series(csv_path)
        render_level_chart(df, title, out_png)
        return df
    except Exception as e:
        print(f"[WARN] skip {csv_path}: {e}")
        return None


def calc_delta_level(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    return float(df["value"].iloc[-1] - df["value"].iloc[0])


def compute_intraday_pct(csv_intraday: str) -> tuple[float | None, str | None, str | None]:
    """
    Compute simple percent for the day using first row as 'open' and last row as 'close':
        pct = (close - open) / abs(open) * 100
    Returns (pct_float, start_iso, end_iso)
    """
    if not os.path.exists(csv_intraday):
        return None, None, None
    try:
        dfi = load_series(csv_intraday)
        if dfi.empty:
            return None, None, None
        op = float(dfi["value"].iloc[0])
        cl = float(dfi["value"].iloc[-1])
        if op == 0:
            return None, dfi["ts"].iloc[0].isoformat(), dfi["ts"].iloc[-1].isoformat()
        pct = (cl - op) / abs(op) * 100.0
        return float(pct), dfi["ts"].iloc[0].isoformat(), dfi["ts"].iloc[-1].isoformat()
    except Exception:
        return None, None, None


def write_stats(delta_level: float, pct_1d: float | None):
    data = {
        "index_key": INDEX_KEY,
        "pct_1d": round(pct_1d, 4) if isinstance(pct_1d, (int, float)) else None,
        "delta_level": round(delta_level, 6) if isinstance(delta_level, (int, float)) else None,
        "scale": "level",
        "basis": "n/a",   # charts are level; percent basis not applicable
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[OK] wrote {STATS_JSON}")


def write_intraday_post(pct_1d: float | None, start_iso: str | None, end_iso: str | None):
    """
    For X posting. If pct_1d is None, write A%=N/A.
    """
    if pct_1d is None:
        line = f"{INDEX_KEY.upper()} 1d: A%=N/A (basis n/a)"
    else:
        sign = "+" if pct_1d >= 0 else ""
        # display with 2 decimals
        line = f"{INDEX_KEY.upper()} 1d: A%={sign}{pct_1d:.2f}% (basis open {start_iso}->{end_iso})"
    with open(POST_INTRADAY, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[OK] wrote {POST_INTRADAY}: {line}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) charts (level)
    df_1d = try_render(CSV_1D, f"{INDEX_KEY.upper()} (1d)", PNG_1D)
    df_7d = try_render(CSV_7D, f"{INDEX_KEY.upper()} (7d)", PNG_7D)
    df_1m = try_render(CSV_1M, f"{INDEX_KEY.upper()} (1m)", PNG_1M)
    df_1y = try_render(CSV_1Y, f"{INDEX_KEY.upper()} (1y)", PNG_1Y)

    # delta from 1d (level)
    delta = calc_delta_level(df_1d)

    # 2) intraday percent for post
    pct, start_iso, end_iso = compute_intraday_pct(CSV_INTRADAY)
    write_intraday_post(pct, start_iso, end_iso)

    # 3) stats json (level + pct_1d for convenience)
    write_stats(delta, pct)


if __name__ == "__main__":
    main()
