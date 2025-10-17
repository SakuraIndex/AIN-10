#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1日騰落率（open -> latest）だけを計算して、
- docs/outputs/<index>_post_intraday.txt
- docs/outputs/<index>_stats.json
を“唯一のスクリプト”として上書きする。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, time, timezone

import pandas as pd


def detect_cols(df: pd.DataFrame) -> tuple[str, str]:
    lower = {c: c.lower() for c in df.columns}
    t_candidates = ["datetime", "time", "timestamp", "date", "dt"]
    v_candidates = ["value", "y", "index", "score", "close", "price"]

    # AIN-10 列名等の特殊ケース（ハイフン除去で判定）
    for c in df.columns:
        if lower[c].replace("-", "") in ("ain10", "ain"):
            v_candidates.insert(0, c)

    tcol = None
    for k in t_candidates:
        for c in df.columns:
            if lower[c] == k:
                tcol = c
                break
        if tcol:
            break
    if tcol is None:
        tcol = df.columns[0]

    vcol = None
    for k in v_candidates:
        for c in df.columns:
            if lower[c] == k or c == k:
                vcol = c
                break
        if vcol:
            break
    if vcol is None:
        vcol = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return tcol, vcol


def pick_session_open_last(df: pd.DataFrame) -> tuple[float | None, float | None, str, str]:
    tcol, vcol = detect_cols(df)
    x = df[[tcol, vcol]].dropna().copy()
    x[tcol] = pd.to_datetime(x[tcol], errors="coerce", utc=False)
    x[vcol] = pd.to_numeric(x[vcol], errors="coerce")
    x = x.dropna().sort_values(by=tcol)

    # 市場時間（必要に応じて調整）
    start_t, end_t = time(9, 30), time(15, 50)
    sx = x[(x[tcol].dt.time >= start_t) & (x[tcol].dt.time <= end_t)]
    if sx.empty:
        return None, None, "n/a", "n/a"

    first_ts = sx.iloc[0][tcol]
    last_ts = sx.iloc[-1][tcol]
    first = float(sx.iloc[0][vcol])
    last = float(sx.iloc[-1][vcol])
    return first, last, str(first_ts), str(last_ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    out_json_path = Path(args.out_json)
    out_text_path = Path(args.out_text)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_text_path.parent.mkdir(parents=True, exist_ok=True)

    # 既定の（N/A）テンプレ
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    post_line = f"{args.index_key.upper()} 1d: A%=N/A (basis=n/a valid=n/a)"
    stats = {
        "index_key": args.index_key,
        "pct_1d": None,
        "delta_level": None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": now_iso,
    }

    try:
        df = pd.read_csv(args.csv)
        if not df.empty:
            o, l, fts, lts = pick_session_open_last(df)
            if o is not None and l is not None and o != 0.0:
                pct = (l - o) / o * 100.0
                post_line = f"{args.index_key.upper()} 1d: A%={pct:+.2f}% (basis=open valid={fts} -> {lts})"
                stats.update({"pct_1d": pct, "basis": "open"})
    except Exception as e:
        # 何があっても出力は必ず残す（N/Aでもよい）
        print(f"[warn] percent calc failed: {e}")

    # 上書き出力
    out_text_path.write_text(post_line + "\n", encoding="utf-8")
    out_json_path.write_text(json.dumps(stats), encoding="utf-8")
    print("[ok] post & stats written")

if __name__ == "__main__":
    main()
