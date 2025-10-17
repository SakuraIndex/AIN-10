#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

INDEX_KEY = "ain10"
OUTDIR = Path("docs/outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# --- visual ---
FIG_BG = "#0e0f13"
AX_BG = "#0b0c10"
GRID = "#2a2e3a"
LINE = "#ff6b6b"
FG = "#e7ecf1"
plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.titlecolor": FG,
})

def _load_df():
    csv = INTRADAY_CSV if INTRADAY_CSV.exists() else HISTORY_CSV
    df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

def _plot(s: pd.Series, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.plot(s.index, s.values, color=LINE, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG)
    plt.close(fig)

def compute_delta_and_pct(s: pd.Series):
    if len(s) < 2:
        return None, None
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    delta = last - first
    denom = max(abs(first), abs(last), 1e-6)
    pct = 100.0 * delta / denom
    return delta, pct

def write_outputs():
    df = _load_df()
    s = df[df.columns[-1]]
    delta, pct = compute_delta_and_pct(s)
    start, end = s.index[0], s.index[-1]
    nowz = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # --- JSON ---
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": pct,
        "delta_level": delta,
        "scale": "level",
        "basis": "absmax",
        "updated_at": nowz,
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # --- TXT ---
    line = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level) "
        f"A%={pct:+.2f}% (basis=absmax valid={start}->{end})"
    )
    (OUTDIR / f"{INDEX_KEY}_post_intraday.txt").write_text(line + "\n", encoding="utf-8")

def gen_charts():
    df = _load_df()
    s = df[df.columns[-1]]
    _plot(s.tail(1000), OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _plot(s.tail(7*1000), OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _plot(s.tail(30*1000), OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _plot(s, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

if __name__ == "__main__":
    gen_charts()
    write_outputs()
