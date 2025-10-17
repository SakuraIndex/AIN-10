#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd


def iso_now() -> str:
    """常にUTCのISO8601（Z付き）を返す。"""
    ts = pd.Timestamp.utcnow()  # naive
    # tz-aware なら convert、naive なら localize
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def read_1d(csv_path: Path) -> pd.DataFrame:
    """docs/outputs/*_1d.csv を読み、ts/val の2列に正規化して昇順にする。"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df


def percent_change(first: float, last: float) -> float | None:
    """NaNや0割防止付きの単純騰落率(%単位)。"""
    try:
        if first is None or last is None:
            return None
        if pd.isna(first) or pd.isna(last):
            return None
        if abs(float(first)) < 1e-9:  # 0割回避
            return None
        return (float(last) - float(first)) / abs(float(first)) * 100.0
    except Exception:
        return None


def first_nonzero_row(df: pd.DataFrame, col: str = "val", eps: float = 1e-9):
    """val が 0（またはごく小さい）でない最初の行を返す。無ければ None。"""
    nz = df.loc[~pd.isna(df[col]) & (df[col].abs() > eps)]
    return None if nz.empty else nz.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", choices=["open", "prev_close"], default="open")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val: float | None = None
    delta_level: float | None = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        # 1d の最初と最後（レベル差は常にこの2点で計算）
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val = float(last_row["val"])
        delta_level = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        if args.basis == "open":
            # ① 先頭が 0（ほぼ0）なら、その日の最初の非ゼロ値を基準に%計算
            basis_row = (
                first_row if abs(first_val) > 1e-9 else first_nonzero_row(df, "val")
            )
            if basis_row is not None:
                basis_val = float(basis_row["val"])
                pct_val = percent_change(basis_val, last_val)
                basis_note = (
                    "open" if basis_row.name == first_row.name
                    else f"open(nonzero@{basis_row['ts']:%H:%M})"
                )
            else:
                basis_note = "open(nonzero=N/A)"  # 1d 期間内が全て 0/NaN の場合
                pct_val = None
        else:
            # prev_close 指定時（現状は open と同計算。必要になればここを差し替え）
            basis_note = "prev_close"
            pct_val = percent_change(first_val, last_val)

    # --- TXT 出力 ---
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力 ---
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
