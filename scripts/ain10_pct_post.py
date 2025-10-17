#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple

EPS = 1e-6  # 0割・極端値回避のための閾値（レベルが0付近でも暴れないように）

def iso_now() -> str:
    """UTCのISO8601（Z付き）を返す。"""
    ts = pd.Timestamp.utcnow()
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")

def read_1d(csv_path: Path) -> pd.DataFrame:
    """1d CSVの先頭2列（時刻・値）だけを使って読み込む。"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def read_prev_close(history_csv: Path, first_ts: pd.Timestamp) -> Optional[float]:
    """
    history CSV から「first_ts のカレンダー日より前」の最後の終値を返す。
    先頭2列を（日付/時刻, 値）として解釈する。
    """
    if not Path(history_csv).exists():
        return None
    h = pd.read_csv(history_csv)
    if h.shape[1] < 2:
        return None
    d_col, v_col = h.columns[:2]
    # 日付っぽい列をDateに落とす（時刻でもOK）
    h = h.rename(columns={d_col: "dt", v_col: "close"})
    h["dt"] = pd.to_datetime(h["dt"], errors="coerce")
    h = h.dropna(subset=["dt", "close"]).sort_values("dt").reset_index(drop=True)

    # first_ts の日付より「厳密に前」のもの
    mask = h["dt"] < first_ts.normalize()
    if not mask.any():
        return None
    prev_row = h.loc[mask].iloc[-1]
    try:
        return float(prev_row["close"])
    except Exception:
        return None

def first_nonzero_open(df_1d: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, float]]:
    """
    取引序盤にゼロ埋めが混ざっている場合に備えて、
    先頭から順に |val|>=EPS を満たす最初のレコードを返す。
    """
    for i in range(min(len(df_1d), 20)):  # 念のため20本まで探索
        v = float(df_1d.iloc[i]["val"])
        if abs(v) >= EPS:
            return df_1d.iloc[i]["ts"], v
    return None

def percent_change(base: float, last: float) -> Optional[float]:
    """base→last の騰落率（%）。baseが0付近ならNone。"""
    try:
        if base is None or last is None:
            return None
        if abs(float(base)) < EPS:
            return None
        return (float(last) - float(base)) / abs(float(base)) * 100.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--history", required=True, help="docs/outputs/*_history.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument(
        "--basis",
        choices=["prev_close", "open"],
        default="prev_close",
        help="騰落率の基準（推奨: prev_close）",
    )
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: Optional[float] = None
    delta_level: Optional[float] = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_ts  = pd.to_datetime(first_row["ts"])
        last_ts   = pd.to_datetime(last_row["ts"])
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])

        delta_level = last_val - first_val
        valid_note  = f"{first_ts} -> {last_ts}"

        # --- 騰落率の基準を決定 ---
        base_val: Optional[float] = None

        if args.basis == "prev_close":
            prev_close = read_prev_close(Path(args.history), first_ts)
            if prev_close is not None and abs(prev_close) >= EPS:
                base_val = prev_close
                basis_note = "prev_close"
            else:
                # フォールバック：序盤の最初の非ゼロ値をオープン代替とする
                nz = first_nonzero_open(df)
                if nz is not None:
                    nz_ts, base_val = nz
                    basis_note = f"open(nonzero@{nz_ts.strftime('%H:%M')})"
        else:
            # open基準（ただしゼロ回避）
            if abs(first_val) >= EPS:
                base_val = first_val
                basis_note = "open"
            else:
                nz = first_nonzero_open(df)
                if nz is not None:
                    nz_ts, base_val = nz
                    basis_note = f"open(nonzero@{nz_ts.strftime('%H:%M')})"

        # 騰落率の算出
        if base_val is not None:
            pct_val = percent_change(base_val, last_val)

    # --- TXT 出力 ---
    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    line = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(line, encoding="utf-8")

    # --- JSON 出力 ---
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
