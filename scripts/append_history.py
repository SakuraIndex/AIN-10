#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import datetime
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_CSV  = OUT_DIR / f"{INDEX_KEY}_history.csv"
LATEST_TXT= OUT_DIR / f"{INDEX_KEY}_latest.txt"         # 例: 456.78
POST_TXT  = OUT_DIR / f"{INDEX_KEY}_post.txt"           # 任意（存在すれば）
INTRA_CSV = OUT_DIR / f"{INDEX_KEY}_intraday.csv"       # ts,value の形が望ましい

def _diag(msg): print(f"[append] {msg}", flush=True)

def _read_latest_value() -> float | None:
    # 1) latest.txt
    if LATEST_TXT.exists():
        try:
            v = float(LATEST_TXT.read_text().strip())
            _diag(f"use latest.txt = {v}")
            return v
        except Exception:
            pass
    # 2) post.txt
    if POST_TXT.exists():
        try:
            v = float(POST_TXT.read_text().strip())
            _diag(f"use post.txt = {v}")
            return v
        except Exception:
            pass
    # 3) intraday.csv の最終行
    if INTRA_CSV.exists():
        try:
            dfi = pd.read_csv(INTRA_CSV)
            if dfi.shape[1] >= 2:
                # 先頭2列を ts, val と見なす
                dfi = dfi.rename(columns={dfi.columns[0]:"ts", dfi.columns[1]:"val"})
                dfi["ts"]  = pd.to_datetime(dfi["ts"], errors="coerce")
                dfi        = dfi.dropna(subset=["ts","val"]).sort_values("ts")
                if not dfi.empty:
                    v = float(dfi.iloc[-1]["val"])
                    _diag(f"use intraday.csv last = {v}")
                    return v
        except Exception:
            pass
    return None

def _load_history() -> pd.DataFrame:
    if HIST_CSV.exists():
        try:
            df = pd.read_csv(HIST_CSV)
            if df.shape[1] >= 2:
                df = df.rename(columns={df.columns[0]:"date", df.columns[1]:"value"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df.dropna(subset=["date","value"]).drop_duplicates(subset=["date"], keep="last")
                return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=["date","value"])

def main():
    today = datetime.utcnow().date()   # 日付キーはUTC基準（市場TZで管理したい場合は適宜変更）
    val = _read_latest_value()
    if val is None:
        _diag("no source value found; skip append.")
        return

    df = _load_history()
    # 既存日に上書き or 新規行追加
    df = df[df["date"] != today]
    df = pd.concat([df, pd.DataFrame([{"date": today, "value": val}])], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    # 保存（ヘッダ: date,value）
    df_out = df.copy()
    df_out["date"] = df_out["date"].astype(str)
    HIST_CSV.write_text(df_out.to_csv(index=False))
    _diag(f"APPENDED {today} = {val} -> {HIST_CSV}")

if __name__ == "__main__":
    main()
