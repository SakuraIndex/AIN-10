#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長期チャート生成（1d/7d/1m/1y）
- CSVの列名を自動推定（例: ['Datetime', 'AIN-10'] などもOK）
- 画像は level（指数のレベル）線のみを描画（%は描かない）
- ain10_stats.json は 1d% を N/A 固定で出力
- ain10_post_intraday.txt も A%=N/A を出力（X投稿用の文面）
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ======== 設定 ========
OUTDIR = "docs/outputs"

# グリッド・背景など最低限の見た目
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


# ======== CSV ロード（列名の自動推定）========
TIME_CANDIDATES = ["time", "ts", "timestamp", "date", "datetime", "Datetime"]
VALUE_CANDIDATES = ["level", "value", "y", "index", "score", "close", "price"]

def _first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

def load_series(csv_path: str, tcol: Optional[str] = None, vcol: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # 時刻列の推定
    time_col: Optional[str] = tcol if tcol and tcol in df.columns else _first_existing(df, TIME_CANDIDATES)
    if time_col is None:
        # 最初の列を時刻と見なす
        time_col = df.columns[0]

    # 値列の推定
    if vcol and vcol in df.columns:
        val_col = vcol
    else:
        # 既知候補を優先
        val_col = _first_existing(df, VALUE_CANDIDATES)
        if val_col is None:
            # 時刻以外の最初の数値列
            candidates = [c for c in df.columns if c != time_col]
            # まずは数値化を試みる
            for c in candidates:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                # 最後の手段: 時刻以外の2列目を使う
                if len(df.columns) >= 2:
                    val_col = df.columns[1] if df.columns[1] != time_col else df.columns[0]
                else:
                    raise ValueError(f"Could not infer value column in {csv_path}")
            else:
                val_col = num_cols[0]

    # 時刻を datetime に
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df = df.dropna(subset=[time_col])

    # 値を数値に
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    out = df[[time_col, val_col]].rename(columns={time_col: "time", val_col: "level"})
    out = out.sort_values("time").reset_index(drop=True)
    return out


# ======== 描画 ========
def render_level_chart(df: pd.DataFrame, title: str, out_png: str) -> None:
    if df.empty:
        raise ValueError("Empty dataframe, nothing to plot.")

    configure_matplotlib()
    fig, ax = plt.subplots()

    ax.plot(df["time"], df["level"], linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # 余白
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ======== 1d% は N/A で出力 ========
def write_stats(index_key: str, latest_delta_level: Optional[float] = None) -> None:
    """
    pct_1d は N/A 固定。basis='n/a'
    """
    path = os.path.join(OUTDIR, f"{index_key}_stats.json")
    os.makedirs(OUTDIR, exist_ok=True)
    payload = {
        "index_key": index_key,
        "pct_1d": None,
        "delta_level": float(latest_delta_level) if latest_delta_level is not None else None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Wrote {path}")


def write_post_text(index_key: str) -> None:
    """
    X 投稿向け（1d% は N/A）
    """
    path = os.path.join(OUTDIR, f"{index_key}_post_intraday.txt")
    os.makedirs(OUTDIR, exist_ok=True)
    text = f"{index_key.upper()} 1d: A%=N/A (basis n/a)"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Wrote {path}")


# ======== メイン ========
def main():
    parser = argparse.ArgumentParser(description="Generate long-term charts (level only)")
    parser.add_argument("--index-key", default=os.environ.get("INDEX_KEY", "ain10"))
    # 手動で列名指定したいとき用（通常は不要）
    parser.add_argument("--tcol", default=None)
    parser.add_argument("--vcol", default=None)
    args = parser.parse_args()

    index_key = args.index_key

    # 扱うレンジ（CSV は既存のビルドが出す想定）
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

        # 1d のときは最後と最初の level 差分を stats に書くため記録（% は使わない）
        if tag == "1d" and len(df) >= 2:
            latest_delta_level = float(df["level"].iloc[-1] - df["level"].iloc[0])

        render_level_chart(df, title=title, out_png=png_path)
        print(f"Rendered {png_path}")

    # 統計と投稿文言
    write_stats(index_key=index_key, latest_delta_level=latest_delta_level)
    write_post_text(index_key=index_key)


if __name__ == "__main__":
    main()
