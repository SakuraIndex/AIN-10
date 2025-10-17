#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate long-term charts (1d/7d/1m/1y) for AIN-10-like indexes.

✅ ポイント
- テーマ切替え（ダーク/ライト）に対応。既定はダーク。
- 線色はテーマに合わせて自動選択（ダーク=赤、ライト=青）。
- 画像外周の白い縁取りをゼロ化（edgecolor='none'、facecolorを明示）。
- %表記は「レベル系チャートでは常に非表示」。数値は docs/outputs/{index}_stats.json に pct_1d=None として保存。
- 既存 CSV（例: docs/outputs/ain10_1d.csv など）に合わせ、Datetime 列とインデックス列名を自動検知。

使い方（GitHub Actions から直接呼ぶ想定）:
    python scripts/long_charts.py
環境変数:
    INDEX_KEY : 例 'ain10'（必須）
    THEME     : 'dark' or 'light'（省略可・既定 dark）
出力:
    docs/outputs/{index}_1d.png / 7d.png / 1m.png / 1y.png
    docs/outputs/{index}_stats.json（%は null 固定、basis='n/a'）
"""

from __future__ import annotations
import os
import json
import pathlib as p
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------- Config ----------
OUT_DIR = p.Path("docs/outputs")
FREQS = ["1d", "7d", "1m", "1y"]  # 生成対象
# ダークテーマ用の色（ライトテーマ時は後述で差し替え）
DARK_BG   = "#0b0f17"
DARK_GRID = "#2a3242"
DARK_LINE = "#ff6b6b"  # 赤
LIGHT_BG  = "#ffffff"
LIGHT_GRID= "#e6e6e6"
LIGHT_LINE= "#1f77b4"  # 青


# ---------- Utilities ----------
def detect_series_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    入力CSVの列名を自動推定する。
    期待: 時間軸列 と 値列（レベル） を返す
    優先順位:
      time系: ['Datetime','datetime','timestamp','time','date','Date']
      value系: ['value','level','score','close','price'] + 左記が見つからなければ数値列を探索
      特殊: AIN-10系は 'AIN-10' という列名もあり得るので考慮
    """
    cols = list(df.columns)
    time_cands = ["Datetime","datetime","timestamp","time","date","Date"]
    val_cands  = ["value","level","score","close","price","AIN-10"]

    tcol = next((c for c in time_cands if c in cols), None)
    if tcol is None:
        # DatetimeIndex の CSV の場合、index_col=0 で読んでいることもあるので try
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={"index": "Datetime"}, inplace=True)
            tcol = "Datetime"
        else:
            # 先頭列が時刻に解釈できるならそれを使う
            for c in cols:
                try:
                    pd.to_datetime(df[c])
                    tcol = c
                    break
                except Exception:
                    pass
    if tcol is None:
        raise ValueError(f"Time-like column not found in columns={cols}")

    vcol = next((c for c in val_cands if c in cols), None)
    if vcol is None:
        # 数値列を探す
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"Value-like column not found in columns={cols}")
        # 最初の数値列を採用
        vcol = numeric[0]

    return tcol, vcol


def load_csv_for(freq: str, index_key: str) -> pd.DataFrame:
    """
    docs/outputs/{index_key}_{freq}.csv を読み込んで Datetime ソート済み DataFrame を返す。
    """
    path = OUT_DIR / f"{index_key}_{freq}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # index は使わず、列で受けて推定
    df = pd.read_csv(path)
    tcol, vcol = detect_series_columns(df)
    # 時刻にキャストしてソート
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    # 値列は float へ
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    return df[[tcol, vcol]].rename(columns={tcol: "ts", vcol: "level"})


def apply_theme(theme: str) -> Tuple[str, str, str]:
    """
    テーマ設定をmatplotlibへ適用し、(bg, grid, line)色を返す。
    """
    theme = (theme or "dark").lower()
    if theme not in ("dark", "light"):
        theme = "dark"

    if theme == "dark":
        bg, grid, line = DARK_BG, DARK_GRID, DARK_LINE
        matplotlib.rcParams.update({
            "axes.facecolor": DARK_BG,
            "figure.facecolor": DARK_BG,
            "savefig.facecolor": DARK_BG,
            "text.color": "#e6edf3",
            "axes.labelcolor": "#e6edf3",
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9",
            "grid.color": DARK_GRID,
        })
    else:
        bg, grid, line = LIGHT_BG, LIGHT_GRID, LIGHT_LINE
        matplotlib.rcParams.update({
            "axes.facecolor": LIGHT_BG,
            "figure.facecolor": LIGHT_BG,
            "savefig.facecolor": LIGHT_BG,
            "text.color": "#0a0a0a",
            "axes.labelcolor": "#0a0a0a",
            "xtick.color": "#222",
            "ytick.color": "#222",
            "grid.color": LIGHT_GRID,
        })
    # 余白・フォントなど
    matplotlib.rcParams.update({
        "axes.edgecolor": "none",   # 軸外枠も描かない
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlepad": 8,
        "figure.dpi": 120,
    })
    return bg, grid, line


def render_chart(df: pd.DataFrame, title: str, out_png: p.Path,
                 theme: str) -> None:
    """
    レベルチャートを描画して保存。
    - %は表示しない（この関数はレベルのみ）
    - 図外枠の白縁を出さない（edgecolor='none'、facecolor明示、tight+pad最小）
    """
    bg, grid, line = apply_theme(theme)

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    # 背景
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # 線
    ax.plot(df["ts"], df["level"], lw=2.2, color=line, solid_joinstyle="round")

    # ラベル・体裁（%は出さない）
    ax.set_title(title, loc="center", pad=10, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # 目盛間隔（時系列を自然に）
    fig.autofmt_xdate()

    # 外枠（spines）と図全体の白縁を消す
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 余白最小化。bbox_inches='tight' と pad_inches を小さめに
    plt.tight_layout(pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=144,
        facecolor=fig.get_facecolor(),
        edgecolor="none",          # ← 白い線を出さない
        bbox_inches="tight",
        pad_inches=0.08,           # ほぼゼロに
        transparent=False,
    )
    plt.close(fig)


def write_stats(index_key: str) -> None:
    """
    レベル系のため、%は常に N/A（null）。delta_level も intraday 側で扱うためここでは null。
    """
    out = {
        "index_key": index_key,
        "pct_1d": None,        # ← レベル系では X 用の%は別スクリプトで計算; ここは常に None
        "delta_level": None,   # ← レベル差は intraday 投稿側で担う。ここは None
        "scale": "level",
        "basis": "n/a",
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (OUT_DIR / f"{index_key}_stats.json").write_text(json.dumps(out), encoding="utf-8")


def main():
    index_key = os.environ.get("INDEX_KEY", "").strip()
    if not index_key:
        raise SystemExit("Env INDEX_KEY is required (e.g. ain10)")
    theme = os.environ.get("THEME", "dark").strip().lower()

    # 1) CSV を読みつつ、4枚のレベルチャートを生成
    for freq in FREQS:
        try:
            df = load_csv_for(freq, index_key)
        except FileNotFoundError:
            # CSVが無ければスキップ（存在するものだけ描画）
            continue

        title = f"{index_key.upper()} ({freq})"
        out_png = OUT_DIR / f"{index_key}_{freq}.png"
        render_chart(df, title, out_png, theme)

    # 2) レベル系の stats は%をN/A固定で出す
    write_stats(index_key)


if __name__ == "__main__":
    main()
