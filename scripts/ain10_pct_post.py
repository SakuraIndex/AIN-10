#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd

# 分母が小さすぎる時の暴発抑制（long_charts.py と同値）
EPS = 1.0

def iso_now() -> str:
    """UTCのISO8601（Z付き）"""
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")

def read_1d(csv_path: Path) -> pd.DataFrame:
    """docs/outputs/*_1d.csv を読み、[ts,val] に正規化して昇順整列"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

def choose_baseline(df_day: pd.DataFrame, basis: str) -> tuple[float | None, str]:
    """
    ベース値を選ぶ:
      - basis == "open": 当日の最初の値（寄り）をまず使う。|open| < EPS なら fallback
      - basis == "stable@10:00": 10:00以降で最初に |val| >= EPS のもの
      - basis == "auto": open を優先し、ダメなら stable@10:00 に倒す
    最後の最後に「最初の値（条件なし）」で必ず基準を返す（暴発は別関数で抑制）。
    """
    if df_day.empty:
        return None, "no_pct_col"

    # 1) open を試す（auto と open のとき）
    if basis in ("open", "auto"):
        open_val = float(df_day.iloc[0]["val"])
        if abs(open_val) >= EPS:
            return open_val, "open"

    # 2) 10:00 以降の安定点
    mask = (df_day["ts"].dt.hour > 10) | (
        (df_day["ts"].dt.hour == 10) & (df_day["ts"].dt.minute >= 0)
    )
    cand = df_day.loc[mask & (df_day["val"].abs() >= EPS)]
    if not cand.empty:
        return float(cand.iloc[0]["val"]), "stable@10:00"

    # 3) |val| >= EPS の最初の点（時間帯問わず）
    cand2 = df_day.loc[df_day["val"].abs() >= EPS]
    if not cand2.empty:
        return float(cand2.iloc[0]["val"]), "first|val|>=EPS"

    # 4) 最後の最後の fallback：最初の値（条件なし）
    return float(df_day.iloc[0]["val"]), "first_any"

def percent_change(first: float, last: float) -> float | None:
    """
    安全版の%計算:
        (last - first) / max(|first|, |last|, EPS) * 100
    EPS を 1.0 に引き上げ、小さすぎる分母を強制的に底上げ。
    """
    try:
        if first is None or last is None:
            return None
        denom = max(abs(float(first)), abs(float(last)), EPS)
        return (float(last) - float(first)) / denom * 100.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # open を優先しつつダメなら stable@10:00 に倒す "auto" をデフォルト
    ap.add_argument("--basis", choices=["open", "stable10", "auto"], default="auto")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val: float | None = None
    delta_level: float | None = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        # 当日だけに絞る
        day = df["ts"].dt.floor("D").iloc[-1]
        df_day = df[df["ts"].dt.floor("D") == day]
        if not df_day.empty:
            desired_basis = (
                "open"
                if args.basis == "open"
                else ("stable@10:00" if args.basis == "stable10" else "auto")
            )
            baseline, basis_note = choose_baseline(df_day, desired_basis)

            first_ts = df_day.iloc[0]["ts"]
            last_ts = df_day.iloc[-1]["ts"]
            valid_note = f"{first_ts}->{last_ts}"

            last_val = float(df_day.iloc[-1]["val"])
            # “レベル差”は生値で（%ではない）
            delta_level = last_val - float(baseline)

            pct_val = percent_change(baseline, last_val)

    # --- TXT 出力
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力（scale は percent）
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "percent",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
