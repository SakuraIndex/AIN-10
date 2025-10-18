#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

EPS = 1e-6  # divide-by-zero 回避

def iso_now() -> str:
    """UTC の ISO8601(Z) 文字列を返す（tz-aware/naive どちらでも安全）。"""
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
    """当日 10:00±5分で最も近い値。無ければ None。"""
    if df.empty:
        return None
    day = df["ts"].dt.floor("D").max()
    target = day + pd.Timedelta(hours=10)
    win = (df["ts"] >= target - pd.Timedelta(minutes=5)) & (df["ts"] <= target + pd.Timedelta(minutes=5))
    cand = df.loc[win]
    if cand.empty:
        return None
    i = (cand["ts"] - target).abs().idxmin()
    return float(cand.loc[i, "val"])

def stable_denominator_auto(df: pd.DataFrame) -> tuple[float | None, str, float | None]:
    """
    自動判定の分母を返す (denom, basis_note, base_ref)。
    1) stable@10:00（|v|>=1e-3）を優先（base_ref=10:00値）
    2) だめなら当日|val|の中央値（base_ref=初値近傍）
    """
    v10 = pick_10am_value(df)
    if v10 is not None and abs(v10) >= 1e-3:
        return v10, "stable@10:00", v10
    med_abs = float(df["val"].abs().median()) if not df.empty else None
    if med_abs is not None and med_abs > EPS:
        base_ref = float(df.iloc[0]["val"])
        return med_abs, "median_abs@1d", base_ref
    return None, "n/a", None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # 互換のために --basis を受け付ける（auto / open / prev_close / stable@10:00 / median_abs@1d）
    ap.add_argument("--basis", default="auto")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: float | None = None
    delta_level: float | None = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row  = df.iloc[-1]
        first_val = float(first_row["val"])
        last_val  = float(last_row["val"])
        delta_level = last_val - first_val
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        # -------- 分母と基準値の決定 --------
        resolved_basis = "auto"
        denom: float | None = None
        base_ref: float | None = None

        b = (args.basis or "auto").strip().lower()

        if b == "open" or b == "prev_close":
            # ここでは first を開値相当として扱う
            if abs(first_val) > EPS:
                denom = abs(first_val)
                base_ref = first_val
                resolved_basis = b
            else:
                denom, resolved_basis, base_ref = stable_denominator_auto(df)

        elif b == "stable@10:00":
            v10 = pick_10am_value(df)
            if v10 is not None and abs(v10) >= 1e-3:
                denom = abs(v10)
                base_ref = v10
                resolved_basis = "stable@10:00"
            else:
                denom, resolved_basis, base_ref = stable_denominator_auto(df)

        elif b == "median_abs@1d":
            med_abs = float(df["val"].abs().median())
            if med_abs > EPS:
                denom = med_abs
                base_ref = first_val
                resolved_basis = "median_abs@1d"
            else:
                denom, resolved_basis, base_ref = stable_denominator_auto(df)

        else:  # auto / その他
            denom, resolved_basis, base_ref = stable_denominator_auto(df)

        basis_note = resolved_basis

        # -------- 騰落率計算 --------
        if denom is not None and base_ref is not None and abs(denom) > EPS:
            pct_val = (last_val - base_ref) / max(abs(denom), EPS) * 100.0
        else:
            pct_val = None

    # --- TXT 出力
    pct_str   = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
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
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
