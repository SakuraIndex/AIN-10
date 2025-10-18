#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

EPS = 1e-6  # ゼロ割回避の下限

def iso_now() -> str:
    """UTC の ISO8601(Z) を返す（tz-aware/naive どちらでも安全に）。"""
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")

def read_1d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def pick_10am_value(df: pd.DataFrame) -> float | None:
    """当日 10:00±5分の最も近い値（なければ None）"""
    if df.empty:
        return None
    day = df["ts"].dt.floor("D").max()
    target = day + pd.Timedelta(hours=10)
    window = (df["ts"] >= target - pd.Timedelta(minutes=5)) & (df["ts"] <= target + pd.Timedelta(minutes=5))
    cand = df.loc[window]
    if cand.empty:
        return None
    # 10:00 に最も近い
    i = (cand["ts"] - target).abs().idxmin()
    return float(cand.loc[i, "val"])

def stable_denominator(df: pd.DataFrame) -> tuple[float | None, str]:
    """
    分母と basis 文字列を返す。
    1) 10:00 近傍の値 (stable@10:00) を優先
    2) |10:00| が小さすぎる/取得不可 → 当日の |val| の中央値 (median_abs@1d)
    """
    v10 = pick_10am_value(df)
    if v10 is not None and abs(v10) >= 1e-3:  # 小さすぎる10:00は避ける
        return v10, "stable@10:00"
    # フォールバック：当日の絶対値の中央値
    med_abs = float(df["val"].abs().median()) if not df.empty else None
    if med_abs is not None and med_abs > EPS:
        return med_abs, "median_abs@1d"
    return None, "n/a"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: float | None = None
    delta_level: float | None = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        # 開始・終了（可視範囲の最初と最後）
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])
        delta_level = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        # ％分母の決定
        denom, basis_note = stable_denominator(df)
        if denom is not None and abs(denom) > EPS:
            # 10:00（または中央値）からどれだけ動いたか
            # 10:00値が使われた場合は “前場の基準” に対する全日変化率となる
            if basis_note.startswith("stable@10:00"):
                base_ref = pick_10am_value(df)
                if base_ref is None:
                    base_ref = denom
            else:
                base_ref = first_val  # 参考：初値近傍

            pct_val = (last_val - base_ref) / max(abs(denom), EPS) * 100.0
        else:
            pct_val = None

    # --- TXT
    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",         # 元データのスケール
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
