#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_two_col_series(csv_path: Path) -> pd.DataFrame:
    """先頭列=日時, 2列目=値 として読み込む（列名に依存しない）"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV has insufficient columns: {csv_path}")
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], errors="coerce"),
        "val": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df


def safe_pct(numer: float, denom: float) -> float | None:
    if denom is None or denom == 0:
        return None
    try:
        return (numer / denom) * 100.0
    except Exception:
        return None


def compute_basis_value(basis: str, intraday_df: pd.DataFrame,
                        history_path: Path | None) -> tuple[float | None, str]:
    """
    basis を解決して基準値を返す。
    - open: intraday の最初の有効値
    - prev_close: history CSV（任意）から直近クローズ。無ければ None
    - prev_any: 取りうる最も直近の値（intraday先頭でも可）
    """
    if intraday_df.empty:
        return None, "n/a"

    if basis == "open":
        return float(intraday_df["val"].iloc[0]), "open"

    if basis == "prev_any":
        return float(intraday_df["val"].iloc[0]), "prev_any"

    if basis == "prev_close":
        if history_path and Path(history_path).exists():
            try:
                h = read_two_col_series(Path(history_path))
                if not h.empty:
                    return float(h["val"].iloc[-1]), "prev_close"
            except Exception:
                pass
        return None, "prev_close"  # 情報無しでも落とさない

    # 不明指定は open 扱い
    return float(intraday_df["val"].iloc[0]), "open"


def main():
    ap = argparse.ArgumentParser(description="Compute 1d percent & write posts")
    ap.add_argument("--index-key", required=True, dest="index_key")
    ap.add_argument("--csv", required=True, help="intraday 1d CSV path")
    ap.add_argument("--out-json", required=True, dest="out_json")
    ap.add_argument("--out-text", required=True, dest="out_text")
    ap.add_argument("--basis", choices=["open", "prev_close", "prev_any"],
                    default="open", help="percent basis (default: open)")
    ap.add_argument("--history", default=None,
                    help="optional history CSV (for prev_close)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = read_two_col_series(csv_path)
    updated_at = now_iso()

    # デフォルト（計算不能時）
    pct_1d = None
    delta_level = None
    valid_str = "n/a"

    if not df.empty:
        first_ts = df["ts"].iloc[0]
        last_ts = df["ts"].iloc[-1]
        last_val = float(df["val"].iloc[-1])

        basis_val, basis_used = compute_basis_value(args.basis, df,
                                                    Path(args.history) if args.history else None)
        if basis_val is not None:
            delta_level = last_val - basis_val
            pct_1d = safe_pct(delta_level, basis_val)
            valid_str = f"{first_ts.strftime('%Y-%m-%d %H:%M')}->{last_ts.strftime('%Y-%m-%d %H:%M')}"
        else:
            basis_used = args.basis

    # JSON 出力
    out_json = {
        "index_key": args.index_key,
        "pct_1d": float(pct_1d) if pct_1d is not None else None,
        "delta_level": float(delta_level) if delta_level is not None else None,
        "scale": "level",
        "basis": args.basis,
        "updated_at": updated_at,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False)

    # テキスト出力
    def fmt_pct(p):
        return f"{p:+.2f}%" if p is not None else "N/A"

    def fmt_delta(d):
        return f"{d:+.6f}" if d is not None else "N/A"

    line = (f"{args.index_key.upper()} 1d: Δ={fmt_delta(delta_level)} (level) "
            f"A%={fmt_pct(pct_1d)} (basis={args.basis} valid={valid_str})")

    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(line + "\n")

    print("[OK] wrote:", args.out_json, "and", args.out_text)


if __name__ == "__main__":
    main()
