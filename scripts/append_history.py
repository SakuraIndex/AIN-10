#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10: append one daily row into docs/outputs/ain10_history.csv
- Prefer chaining by daily pct change if available
- Otherwise, use snapshot level but auto-rescale to match yesterday's magnitude
- Timezone: America/New_York, session 09:30–16:00
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

OUTDIR = Path("docs/outputs")
HIST   = OUTDIR / "ain10_history.csv"
STATS  = OUTDIR / "ain10_stats.json"     # 騰落率（%）
SNAP   = OUTDIR / "ain10_latest.txt"     # 直近スナップ（レベル想定）
LAST_RUN = OUTDIR / "_last_run.txt"

NY = pytz.timezone("America/New_York")
SESSION_START = (9, 30)
SESSION_END   = (16, 0)

MAX_ABS_DAILY_MOVE = 0.25    # 25% を超える変化は異常扱い
ROUND = 6

def now_ny() -> datetime:
    return datetime.now(tz=NY)

def today_ymd_ny() -> str:
    return now_ny().strftime("%Y-%m-%d")

def read_history() -> pd.DataFrame:
    if not HIST.exists():
        return pd.DataFrame(columns=["date","value"])
    df = pd.read_csv(HIST)
    # 列を強制固定
    cols = [c.strip().lower() for c in df.columns]
    if "date" not in cols or "value" not in cols:
        return pd.DataFrame(columns=["date","value"])
    df = df.rename(columns={cols[i]:["date","value"][["date","value"].index(cols[i])] if cols[i] in ("date","value") else cols[i] for i in range(len(cols))})
    df = df[["date","value"]]
    return df

def write_history(df: pd.DataFrame) -> None:
    out = df.copy()
    out = out.dropna(subset=["date","value"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").round(ROUND)
    out = out.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    out.to_csv(HIST, index=False)

def read_stats_pct() -> float | None:
    if not STATS.exists():
        return None
    try:
        d = json.loads(STATS.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 代表キーを探す
    for k in ("pct_close","pct_intraday","pct_change","rtn_pct","change_pct"):
        if k in d and isinstance(d[k], (int, float)):
            v = float(d[k])
            # 値が±3未満なら「比率（1.12%=0.0112）」の可能性は低いが、一応判定
            if abs(v) <= 3.0 and abs(v) <= 1.2:   # 典型的に 0.75 などは比率
                return v
            # 100 を超えない通常の % とみなす
            return v / 100.0
    return None

def read_snapshot_level() -> float | None:
    if not SNAP.exists():
        return None
    try:
        x = float(SNAP.read_text(encoding="utf-8").strip())
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None

def session_closed_ny(ts: datetime) -> bool:
    h, m = ts.hour, ts.minute
    sh, sm = SESSION_START
    eh, em = SESSION_END
    return (h > eh) or (h == eh and m >= em)

def rescale_to_match(prev: float, new_abs: float) -> float:
    """桁ズレ補正: new_abs * (10^k) で prev に最も近い倍率を選ぶ"""
    if not (math.isfinite(prev) and math.isfinite(new_abs)) or prev <= 0 or new_abs <= 0:
        return new_abs
    best = new_abs
    best_diff = float("inf")
    for k in (-3,-2,-1,0,1,2,3):
        cand = new_abs * (10 ** k)
        diff = abs(cand - prev)
        if diff < best_diff:
            best, best_diff = cand, diff
    return best

def append_once():
    # 場が閉じるまではスキップ（NY）
    if not session_closed_ny(now_ny()):
        print("[append] NY session not closed yet. skip.")
        return

    df = read_history()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    d_today = pd.to_datetime(today_ymd_ny())
    if not df.empty and (df["date"].dt.date == d_today.date()).any():
        print("[append] already appended today. skip.")
        return

    prev_level = float(df.iloc[-1]["value"]) if not df.empty else None

    # 1) 騰落率（優先）
    pct = read_stats_pct()
    new_level = None
    if pct is not None and prev_level is not None:
        if abs(pct) > MAX_ABS_DAILY_MOVE:
            print(f"[append] abnormal pct={pct*100:.2f}% > {MAX_ABS_DAILY_MOVE*100:.0f}%. skip.")
            return
        new_level = prev_level * (1.0 + pct)
        print(f"[append] by pct: prev={prev_level:.6f} -> new={new_level:.6f} ({pct*100:.2f}%)")
    else:
        # 2) スナップショット（レベル）で補完
        snap = read_snapshot_level()
        if snap is None:
            print("[append] no pct and no snapshot. nothing to do.")
            return
        if prev_level is None:
            new_level = snap
            print(f"[append] init by snapshot: {new_level:.6f}")
        else:
            new_level = rescale_to_match(prev_level, snap)
            move = (new_level - prev_level) / prev_level
            if abs(move) > MAX_ABS_DAILY_MOVE:
                print(f"[append] abnormal jump after rescale: {move*100:.2f}%. skip.")
                return
            print(f"[append] by snapshot-rescale: prev={prev_level:.6f} -> snap={snap:.6f} -> new={new_level:.6f}")

    if not math.isfinite(new_level):
        print("[append] new_level is not finite. skip.")
        return

    # 追記（未来日は拒否）
    if not df.empty and d_today.date() < df.iloc[-1]["date"].date():
        print("[append] today < last date in history. skip.")
        return

    row = pd.DataFrame([{"date": d_today.strftime("%Y-%m-%d"), "value": round(float(new_level), ROUND)}])
    df_out = pd.concat([df[["date","value"]], row], ignore_index=True)
    write_history(df_out)
    LAST_RUN.write_text(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(f"[ok] appended {row.iloc[0].to_dict()}")

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    append_once()
