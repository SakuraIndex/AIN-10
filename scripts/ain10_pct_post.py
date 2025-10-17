#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import time as dtime

import numpy as np
import pandas as pd


def load_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 日時検出
    for c in ["Datetime", "datetime", "timestamp", "time", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.rename(columns={c: "time"})
            break
    else:
        raise ValueError("time column not found in CSV")

    # 値列は最初の数値列を採用（なければ代表候補を数値化）
    num_cols = [c for c in df.columns if c != "time" and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        for c in ["AIN-10", "AIN10", "value", "score", "close", "price", "index"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                num_cols = [c]
                break
    if not num_cols:
        raise ValueError("value column not found in CSV")

    df = df[["time", num_cols[0]]].rename(columns={num_cols[0]: "value"})
    df = df.sort_values("time").reset_index(drop=True)
    return df


def pick_first_last_of_session(df: pd.DataFrame, session: str):
    """
    指定セッション（例 "09:30-15:50"）の当日データから first/last を返す。
    セッションに1件も無ければ、その日の最初/最後をフォールバックとして返す。
    なければ (None, None)。
    """
    s_open, s_close = session.split("-")
    hh, mm = map(int, s_open.split(":"))
    open_t = dtime(hh, mm)
    hh, mm = map(int, s_close.split(":"))
    close_t = dtime(hh, mm)

    if df.empty:
        return None, None, None

    latest_day = df["time"].dt.date.iloc[-1]
    day_df = df[df["time"].dt.date == latest_day].copy()

    if day_df.empty:
        return None, None, latest_day

    in_sess = day_df[
        (day_df["time"].dt.time >= open_t) & (day_df["time"].dt.time <= close_t)
    ]
    target = in_sess if not in_sess.empty else day_df

    first = target.iloc[0]
    last = target.iloc[-1]
    return first, last, latest_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--session", default="09:30-15:50")
    args = ap.parse_args()

    df = load_series(Path(args.csv))

    first, last, the_date = pick_first_last_of_session(df, args.session)

    basis = "n/a"
    pct_1d = None

    if first is not None and last is not None:
        open_val = float(first["value"])
        last_val = float(last["value"])
        if np.isfinite(open_val) and np.isfinite(last_val) and abs(open_val) > 1e-12:
            pct_1d = (last_val - open_val) / abs(open_val) * 100.0
            basis = "open"

    # 投稿テキスト
    if pct_1d is None:
        post_line = f"{args.index_key.upper()} 1d: A%=N/A (basis={basis} valid=n/a)"
    else:
        sign = "+" if pct_1d >= 0 else ""
        date_str = str(the_date)
        post_line = (
            f"{args.index_key.upper()} 1d: A%={sign}{pct_1d:.2f}% "
            f"(basis={basis} valid={date_str} first->latest)"
        )

    # 保存
    Path(args.out_text).write_text(post_line + "\n", encoding="utf-8")

    stats_obj = {
        "index_key": args.index_key,
        "pct_1d": None if pct_1d is None else float(pct_1d),
        "delta_level": None,         # レベル差はサイト側で使わないので固定
        "scale": "level",
        "basis": basis,
        "updated_at": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    Path(args.out_json).write_text(json.dumps(stats_obj, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
