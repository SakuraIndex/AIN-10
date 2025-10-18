#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import re
from typing import Optional, Tuple

import pandas as pd


# -------------------------------------------------------------------
# Time: always UTC ISO8601 (trailing Z). Safe for tz-naive/aware.
# -------------------------------------------------------------------
def iso_now() -> str:
    """常にUTCのISO8601（Z付き）を返す（tz-naive/aware 両対応）"""
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


# -------------------------------------------------------------------
# CSV loader: first two cols -> ts, val
# -------------------------------------------------------------------
def read_1d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Percent change with guards (returns % value, not fraction)
# -------------------------------------------------------------------
def percent_change(first: float, last: float) -> Optional[float]:
    """NaNやゼロ割を避けた単純騰落率（%）"""
    try:
        if first is None or last is None:
            return None
        f = float(first)
        l = float(last)
        if abs(f) < 1e-9:
            return None
        return (l - f) / abs(f) * 100.0
    except Exception:
        return None


# -------------------------------------------------------------------
# Basis parsing & baseline selection
#   - "open"               : その日の先頭サンプル
#   - "prev_close"         : データが無ければ同等に扱う（将来差し替え前提）
#   - "stable@HH:MM"       : 指定時刻以降の最初のサンプル
#   - "nonzero@HH:MM"      : 指定時刻以降で |val|>1e-9 の最初のサンプル
#   - その他（不明・未対応）: baseline 不決定
# -------------------------------------------------------------------
_TIME_RE = re.compile(r"^(?:stable|nonzero)@(\d{1,2}):(\d{2})$")


def pick_baseline(df: pd.DataFrame, basis: str) -> Tuple[Optional[float], str]:
    basis = (basis or "").strip().lower()

    if df.empty:
        return None, "no_data"

    # 1) open
    if basis == "open":
        first_val = float(df.iloc[0]["val"])
        return first_val, "open"

    # 2) prev_close（今は前日終値が無いので open 相当）
    if basis == "prev_close":
        first_val = float(df.iloc[0]["val"])
        return first_val, "prev_close"

    # 3) stable@HH:MM / nonzero@HH:MM
    m = _TIME_RE.match(basis)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        # その日の指定時刻（ローカル想定、df["ts"]は naive or tz-aware でも比較は同日内でOK）
        day = df["ts"].dt.floor("D").min()
        t0 = day + pd.Timedelta(hours=hh, minutes=mm)

        sub = df[df["ts"] >= t0]
        if sub.empty:
            return None, f"{basis}(no_rows)"

        if basis.startswith("nonzero@"):
            # 最初に |val|>1e-9 を探す
            nz = sub[abs(sub["val"]) > 1e-9]
            if nz.empty:
                return None, f"{basis}(no_nonzero)"
            return float(nz.iloc[0]["val"]), f"{basis}"
        else:
            # stable@ は最初のサンプルを採用
            return float(sub.iloc[0]["val"]), f"{basis}"

    # 4) 未対応
    return None, "no_pct_col"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # basis は柔軟指定（choicesは使わない）: open / prev_close / stable@HH:MM / nonzero@HH:MM
    ap.add_argument("--basis", default="open")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: Optional[float] = None
    delta_level: Optional[float] = None
    basis_note: str = "n/a"
    valid_note: str = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        first_val_day = float(first_row["val"])
        last_val = float(last_row["val"])
        delta_level = last_val - first_val_day
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        baseline, basis_note = pick_baseline(df, args.basis)
        if baseline is not None:
            pct_val = percent_change(baseline, last_val)
        else:
            pct_val = None  # A%=N/A

    # --- TXT 出力
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
