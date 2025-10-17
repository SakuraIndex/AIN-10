#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長期チャート生成（1d/7d/1m/1y）
- 旧CLI (--csv/--out/--title) と新CLI（引数なし・一括）の両対応
- CSV列名は自動推定（例: ['Datetime','AIN-10'] などでもOK）
- チャートは level（指数レベル）のみ描画（%は一切表示しない）
- *_stats.json は pct_1d を N/A 固定（basis="n/a"）
- *_post_intraday.txt は X投稿用に A%=N/A を出力
"""

import os
import json
import argparse
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "docs/outputs"

# ---- 見た目 ----
def configure_matplotlib():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#CCCCCC",
        "text.color": "#DDDDDD",
        "xtick.color": "#BBBBBB",
        "ytick.color": "#BBBBBB",
        "savefig.facecolor": "#111111",
        "figure.facecolor": "#111111",
    })

# ---- CSV ロード（列名の自動推定）----
TIME_CANDIDATES = ["time", "ts", "timestamp", "date", "datetime", "Datetime"]
VALUE_CANDIDATES = ["level", "value", "y", "index", "score", "close", "price"]

def _first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        c = lower.get(n.lower())
        if c is not None:
            return c
    return None

def load_series(csv_path: str, tcol: Optional[str] = None, vcol: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # 時刻列
    time_col = tcol if (tcol and tcol in df.columns) else _first_existing(df, TIME_CANDIDATES)
    if time_col is None:
        time_col = df.columns[0]

    # 値列
    if vcol and vcol in df.columns:
        val_col = vcol
    else:
        val_col = _first_existing(df, VALUE_CANDIDATES)
        if val_col is None:
            # 数値列から推定
            candidates = [c for c in df.columns if c != time_col]
            for c in candidates:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                if len(df.columns) >= 2:
                    val_col = df.columns[1] if df.columns[1] != time_col else df.columns[0]
                else:
                    raise ValueError(f"Could not infer value column in {csv_path}")
            else:
                val_col = num_cols[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df = df.dropna(subset=[time_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    out = df[[time_col, val_col]].rename(columns={time_col: "time", val_col: "level"})
    out = out.sort_values("time").reset_index(drop=True)
    return out

# ---- 描画 ----
def render_level_chart(df: pd.DataFrame, title: str, out_png: str) -> None:
    if df.empty:
        raise ValueError("Empty dataframe.")
    configure_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["level"], linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---- stats / post ----
def write_stats(index_key: str, latest_delta_level: Optional[float]) -> None:
    path = os.path.join(OUTDIR, f"{index_key}_stats.json")
    os.makedirs(OUTDIR, exist_ok=True)
    payload = {
        "index_key": index_key,
        "pct_1d": None,  # N/A 固定
        "delta_level": float(latest_delta_level) if latest_delta_level is not None else None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Wrote {path}")

def write_post_text(index_key: str) -> None:
    path = os.path.join(OUTDIR, f"{index_key}_post_intraday.txt")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{index_key.upper()} 1d: A%=N/A (basis n/a)\n")
    print(f"Wrote {path}")

# ---- メイン ----
def main():
    parser = argparse.ArgumentParser(description="Generate long-term charts (level only)")
    parser.add_argument("--index-key", default=os.environ.get("INDEX_KEY", "ain10"))

    # 旧インターフェイス（単発実行用）
    parser.add_argument("--csv", default=None, help="(legacy) input csv path")
    parser.add_argument("--out", default=None, help="(legacy) output png path")
    parser.add_argument("--title", default=None, help="(legacy) chart title")
    parser.add_argument("--tcol", default=None, help="(legacy/opt) time column name")
    parser.add_argument("--vcol", default=None, help="(legacy/opt) value/level column name")

    args = parser.parse_args()
    index_key = args.index_key

    # 1) 旧式: --csv / --out が来たら単発レンダリング
    if args.csv and args.out:
        df = load_series(args.csv, tcol=args.tcol, vcol=args.vcol)
        render_level_chart(df, title=args.title or index_key.upper(), out_png=args.out)

        # 単発でも stats/post は更新（pct_1d はN/Aのまま）
        delta_level = None
        if len(df) >= 2:
            delta_level = float(df["level"].iloc[-1] - df["level"].iloc[0])
        write_stats(index_key, delta_level)
        write_post_text(index_key)
        return

    # 2) 新式: 一括生成（1d/7d/1m/1y）
    ranges = [
        ("1d", f"{index_key}_1d.csv", f"{index_key}_1d.png", f"{index_key.upper()} (1d)"),
        ("7d", f"{index_key}_7d.csv", f"{index_key}_7d.png", f"{index_key.upper()} (7d)"),
        ("1m", f"{index_key}_1m.csv", f"{index_key}_1m.png", f"{index_key.upper()} (1m)"),
        ("1y", f"{index_key}_1y.csv", f"{index_key}_1y.png", f"{index_key.upper()} (1y)"),
    ]

    latest_delta_level = None
    for tag, csv_name, png_name, title in ranges:
        csv_path = os.path.join(OUTDIR, csv_name)
        png_path = os.path.join(OUTDIR, png_name)
        if not os.path.exists(csv_path):
            print(f"Skip {tag}: {csv_path} not found")
            continue
        df = load_series(csv_path, tcol=args.tcol, vcol=args.vcol)
        if tag == "1d" and len(df) >= 2:
            latest_delta_level = float(df["level"].iloc[-1] - df["level"].iloc[0])
        render_level_chart(df, title=title, out_png=png_path)
        print(f"Rendered {png_path}")

    write_stats(index_key, latest_delta_level)
    write_post_text(index_key)

if __name__ == "__main__":
    main()
