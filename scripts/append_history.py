#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIN-10: append one daily row into docs/outputs/ain10_history.csv
- Prefer chaining by daily pct change if available
- Otherwise, use snapshot level but auto-rescale to match yesterday's magnitude
- Timezone: America/New_York, session 09:30–16:00
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytz

OUTDIR = Path("docs/outputs")
HIST   = OUTDIR / "ain10_history.csv"
STATS  = OUTDIR / "ain10_stats.json"     # 騰落率が取れるならここから
SNAP   = OUTDIR / "ain10_latest.txt"     # 直近スナップ（絶対値）
LAST_RUN = OUTDIR / "_last_run.txt"

NY = pytz.timezone("America/New_York")
SESSION_START = (9, 30)
SESSION_END   = (16, 0)

MAX_ABS_DAILY_MOVE = 0.25    # 25% を超える変化は異常としてスキップ

def now_ny():
    return datetime.now(tz=NY)

def today_ymd_ny():
    return now_ny().strftime("%Y-%m-%d")

def read_history() -> pd.DataFrame:
    if not HIST.exists():
        return pd.DataFrame(columns=["date","value"])
    df = pd.read_csv(HIST)
    if df.shape[1] < 2:
        df = pd.DataFrame(columns=["date","value"])
    df.columns = ["date","value"]
    return df

def write_history(df: pd.DataFrame):
    df.to_csv(HIST, index=False)

def read_stats_pct() -> float | None:
    # 期待キー: {"pct_intraday": 1.23, ...} / あるいは {"pct_close": 1.23}
    if not STATS.exists():
        return None
    try:
        d = json.loads(STATS.read_text(encoding="utf-8"))
        for k in ("pct_close", "pct_intraday", "pct_change"):
            if k in d and isinstance(d[k], (int, float)):
                return float(d[k]) / 100.0
    except Exception:
        pass
    return None

def read_snapshot_level() -> float | None:
    if not SNAP.exists():
        return None
    try:
        return float(SNAP.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def is_trading_done_ny(ts: datetime) -> bool:
    h, m = ts.hour, ts.minute
    sh, sm = SESSION_START
    eh, em = SESSION_END
    after_open   = (h > sh) or (h == sh and m >= sm)
    after_close  = (h > eh) or (h == eh and m >= em)
    return after_close

def rescale_to_match(prev: float, new_abs: float) -> float:
    """桁ズレ補正: new_abs * (10^k) で prev に最も近い倍率を選ぶ"""
    if prev <= 0 or new_abs <= 0:
        return new_abs
    candidates = []
    for k in (-2, -1, 0, 1, 2):
        cand = new_abs * (10 ** k)
        diff = abs(cand - prev)
        candidates.append((diff, cand))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def append_once():
    # まだ場が終わっていないなら何もしない（NYクローズ後に追記）
    if not is_trading_done_ny(now_ny()):
        print("[append] NY session not closed yet. skip.")
        return

    df = read_history()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 今日の日付（NY）
    d_today = pd.to_datetime(today_ymd_ny())

    # 既に今日分が入っていれば何もしない
    if not df.empty and (df["date"].dt.date == d_today.date()).any():
        print("[append] already appended today. skip.")
        return

    # 前日レベル
    if df.empty:
        prev_level = None
    else:
        prev_level = float(df.iloc[-1]["value"])

    # まずは pct を優先
    pct = read_stats_pct()
    if pct is not None and prev_level is not None:
        if abs(pct) > MAX_ABS_DAILY_MOVE:
            print(f"[append] abnormal pct={pct*100:.2f}% > {MAX_ABS_DAILY_MOVE*100:.0f}%. skip.")
            return
        new_level = prev_level * (1.0 + pct)
        print(f"[append] by pct: prev={prev_level:.6f} -> new={new_level:.6f} ({pct*100:.2f}%)")
    else:
        # スナップショットで補完（桁補正）
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

    # 追記
    row = pd.DataFrame([{"date": d_today.strftime("%Y-%m-%d"), "value": new_level}])
    df_out = pd.concat([df, row], ignore_index=True)
    write_history(df_out)
    LAST_RUN.write_text(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    append_once()
