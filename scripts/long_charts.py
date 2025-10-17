#!/usr/bin/env python3
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 共通設定 ---------------------------------------------------------------
OUT_DIR = Path("docs/outputs")
DARK_BG = "#0d1117"   # GitHub ダークに馴染む黒
FG      = "#e6edf3"   # 文字色
GRID    = "#30363d"   # 罫線
RED     = "#ff6b6b"   # 下落トーン（既定の AIN 系は赤で表示）
GREEN   = "#31c48d"   # 上昇トーン

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
})

# ---- CSV 読み込みユーティリティ --------------------------------------------
def load_series(csv_path: Path) -> pd.DataFrame:
    """
    docs/outputs/*_{period}.csv を読み込み、時系列にソート。
    列は以下のいずれかに対応:
      - ["Datetime", "<INDEX_NAME>"]
      - ["time", "value"] など
    値列は「最初の数値列」を採用。
    """
    df = pd.read_csv(csv_path)
    # 日時候補
    for c in ["Datetime", "datetime", "timestamp", "time", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.sort_values(c).reset_index(drop=True)
            df = df.rename(columns={c: "time"})
            break
    else:
        raise ValueError(f"datetime column not found in {csv_path.name}")

    # 値列は最初の数値列
    num_cols = [c for c in df.columns if c != "time" and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        # 数値が1つもない場合は、「AIN-10」「AIN10」「value」等の文字列列を安全に変換
        for c in ["AIN-10", "AIN10", "value", "score", "close", "price", "index"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                num_cols = [c]
                break
    if not num_cols:
        raise ValueError(f"value column not found in {csv_path.name}")

    df = df.rename(columns={num_cols[0]: "value"})
    return df[["time", "value"]]

# ---- プロット ----------------------------------------------------------------
def render_chart(period: str, index_key: str, color: str):
    csv_path = OUT_DIR / f"{index_key}_{period}.csv"
    if not csv_path.exists():
        # CSV がない時はスキップ（ワークフロー上は no-op）
        return

    df = load_series(csv_path)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)

    # 白い外枠を消す（spineを全消し）
    for s in ax.spines.values():
        s.set_visible(False)

    ax.plot(df["time"], df["value"], color=color, linewidth=2.2)

    ax.set_title(f"{index_key.upper()} ({period})", fontsize=16, pad=10, color=FG)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")

    # 余白を調整（白フチ防止）
    plt.margins(x=0.02, y=0.08)
    plt.tight_layout()

    out_png = OUT_DIR / f"{index_key}_{period}.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def main():
    index_key = os.getenv("INDEX_KEY", "ain10")

    # カラーモード（上昇/下落）切り替え：
    # 直近1d の開始値と終了値でざっくり判定（デフォルトは赤）
    color = RED
    try:
        d1 = load_series(OUT_DIR / f"{index_key}_1d.csv")
        if len(d1) >= 2 and np.isfinite(d1["value"].iloc[0]) and np.isfinite(d1["value"].iloc[-1]):
            color = GREEN if (d1["value"].iloc[-1] - d1["value"].iloc[0]) >= 0 else RED
    except Exception:
        pass

    for period in ["1d", "7d", "1m", "1y"]:
        try:
            render_chart(period, index_key, color)
        except Exception as e:
            print(f"[WARN] render {period}: {e}")

if __name__ == "__main__":
    main()
