#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level-only charts for AIN10 (or any index series)
- NEVER print % on the figure
- Minimal, clean title (e.g., "AIN10 (1d)")
- Robust column detection (time & level)
- Works for intraday or daily data alike
- Typical usage:
    python scripts/long_charts.py \
        --csv docs/outputs/ain10_1d.csv \
        --out docs/outputs/ain10_1d.png \
        --title "AIN10 (1d)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Default style ----------
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_DPI = 160
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LINEWIDTH = 2.0
# -----------------------------------


def _guess_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name (case-insensitive) that matches candidates."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_series(csv_path: Path, tcol: Optional[str] = None, vcol: Optional[str] = None) -> pd.DataFrame:
    """
    Load time/value series from CSV. Detect columns if not given.
    Expected columns (case-insensitive):
      - time:  one of ["ts","time","timestamp","date","datetime"]
      - value: one of ["level","value","y","index","score","close","price"]
    Returns DataFrame with columns ["ts","level"] sorted by ts (datetime64).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if tcol is None:
        tcol = _guess_col(df.columns, ["ts", "time", "timestamp", "date", "datetime"])
    if vcol is None:
        vcol = _guess_col(df.columns, ["level", "value", "y", "index", "score", "close", "price"])

    if tcol is None or vcol is None:
        raise ValueError(
            f"Missing required columns in {csv_path}. "
            f"Need time∈[ts,time,timestamp,date,datetime], value∈[level,value,y,index,score,close,price]. "
            f"Found: {list(df.columns)}"
        )

    df = df[[tcol, vcol]].rename(columns={tcol: "ts", vcol: "level"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "level"]).sort_values("ts")
    return df


def render_level_chart(
    csv_path: Path,
    out_png: Path,
    title: str = "AIN10 (1d)",
    tcol: Optional[str] = None,
    vcol: Optional[str] = None,
    figsize=DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> None:
    """
    Render a clean level chart from CSV to PNG.
    - No percent or stats annotations
    - Minimal title
    """
    df = load_series(csv_path, tcol=tcol, vcol=vcol)
    if df.empty:
        raise ValueError(f"No data in {csv_path}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["ts"], df["level"], linewidth=LINEWIDTH)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    ax.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Index (level)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Render level-only chart (no % or stats)")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV path")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default="AIN10 (1d)", help="Figure title")
    p.add_argument("--tcol", type=str, default=None, help="Explicit time column")
    p.add_argument("--vcol", type=str, default=None, help="Explicit value/level column")
    args = p.parse_args()

    render_level_chart(
        csv_path=args.csv,
        out_png=args.out,
        title=args.title,
        tcol=args.tcol,
        vcol=args.vcol,
    )


if __name__ == "__main__":
    main()
