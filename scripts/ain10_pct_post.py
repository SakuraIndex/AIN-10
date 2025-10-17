#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV too few columns: {path}")
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[ts_col, val_col]].rename(columns={ts_col: "ts", val_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def safe_pct(last: float, base: float) -> float | None:
    """安全な比率計算（ゼロ割・極端値を除外）"""
    if base == 0 or not np.isfinite(base) or not np.isfinite(last):
        return None
    pct = (last - base) / abs(base) * 100.0
    if abs(pct) > 25:  # ±25%以上は異常扱いで無視（AIN10の変動レンジ想定）
        return None
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--history", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    df = read_series(Path(args.csv))
    pct = None
    basis = "n/a"
    valid = "n/a"

    if not df.empty:
        open_val = df["value"].iloc[0]
        last_val = df["value"].iloc[-1]
        avg_scale = float(np.nanmean(np.abs(df["value"]))) or 1.0

        # (1) 通常の open 比
        pct = safe_pct(last_val, open_val)
        basis = "open"

        # (2) 異常またはゼロの場合は平均スケールで相対変化率
        if pct is None:
            pct = (last_val - open_val) / avg_scale * 100.0
            basis = "avg"

        # (3) まだ極端なら無効化
        if not np.isfinite(pct) or abs(pct) > 25:
            pct = None
            basis = "n/a"

        valid = f"{str(df['ts'].iloc[0]).split(' ')[0]} first->latest"

    # ---- 出力 ----
    if pct is None:
        pct_str = "N/A"
    else:
        pct_str = f"{pct:+.2f}%"

    text = f"{args.index_key.upper()} 1d: Δ=N/A (level)  A%={pct_str} (basis={basis} valid={valid})"
    Path(args.out_text).write_text(text + "\n", encoding="utf-8")

    stats = {
        "index_key": args.index_key,
        "pct_1d": None if pct is None else round(float(pct), 4),
        "delta_level": None,
        "scale": "level",
        "basis": basis,
        "updated_at": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    Path(args.out_json).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
