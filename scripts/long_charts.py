#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt

JST = timezone(timedelta(hours=9))
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

SMA_WINDOWS = [5, 25, 75]
VOLUME_COLUMN_CANDIDATES = ["volume", "vol", "出来高"]

def log(msg): print(f"[long_charts] {msg}")

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_input(base, key):
    """
    優先: intraday -> history
    無ければ *_intraday.*, *_history.* を自動検出
    """
    cand = first_existing([
        os.path.join(base, f"{key}_intraday.csv"),
        os.path.join(base, f"{key}_intraday.txt"),
        os.path.join(base, f"{key}.csv"),
    ])
    if cand:
        log(f"input(primary): {cand}")
        return cand, "intraday"

    # 自動検出（intraday）
    ii = sorted(glob.glob(os.path.join(base, "*_intraday.csv")) +
                glob.glob(os.path.join(base, "*_intraday.txt")))
    if ii:
        log(f"input(primary-fallback): {ii[0]}")
        return ii[0], "intraday"

    # history フォールバック
    cand = first_existing([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand:
        log(f"input(history): {cand}")
        return cand, "history"

    hh = sorted(glob.glob(os.path.join(base, "*_history.csv")) +
                glob.glob(os.path.join(base, "*_history.txt")))
    if hh:
        log(f"input(history-fallback): {hh[0]}")
        return hh[0], "history"

    return None, None

def read_data(path, kind):
    """CSV/TXTから (time,value,volume) を抽出"""
    df = pd.read_csv(path)
    orig_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # ゆるい列名検出
    t_candidates = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    v_candidates = [c for c in df.columns if any(k in c for k in ["close","price","value","index","終値","値"])]
    vol_candidates = [c for c in df.columns if any(k in c for k in ["volume","vol","出来高"])]

    # 最低限 tcol/vcol を決める
    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = v_candidates[0] if v_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    volcol = vol_candidates[0] if vol_candidates else None

    def parse_time(x):
        if pd.isna(x): return pd.NaT
        s = str(x)
        if re.fullmatch(r"\d{10}", s):  # epoch seconds
            return datetime.fromtimestamp(int(s), tz=JST)
        try:
            t = pd.to_datetime(s)
            if t.tzinfo is None:
                t = t.tz_localize(JST)
            return t.tz_convert(JST)
        except Exception:
            return pd.NaT

    df["time"] = df[tcol].apply(parse_time)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    df["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0

    df = df.dropna(subset=["time","value"]).sort_values("time")

    # historyのとき volume が日次合計想定 / intraday は生値想定
    if kind == "history":
        # すでに日次になっている前提なのでそのまま返す
        return df[["time","value","volume"]].reset_index(drop=True)

    # intraday のときでも、最低限返す
    return df[["time","value","volume"]].reset_index(drop=True)

def to_daily(df, kind):
    """日次変換：intraday→日次(終値/出来高合計), history→そのまま"""
    if kind == "history":
        # 既に日次が多いので time を日付丸めだけ明示
        d = df.copy()
        d["time"] = pd.to_datetime(d["time"]).dt.tz_convert(JST).dt.normalize().dt.tz_localize(JST)
        return d.groupby("time", as_index=False).agg({"value":"last","volume":"sum"}).sort_values("time")

    # intraday → 同一日で最後の値（終値）＋出来高合計
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(JST).dt.date
    daily = d.groupby("date", as_index=False).agg({"value":"last","volume":"sum"})
    daily["time"] = pd.to_datetime(daily["date"]).dt.tz_localize(JST)
    return daily[["time","value","volume"]].sort_values("time")

def plot_chart(df, key, label):
    """価格 + SMA + 出来高。データ1点でもPNG出力"""
    if df.empty:
        log(f"no data for {key}_{label}, write placeholder CSV only")
        df.to_csv(f"docs/outputs/{key}_{label}.csv", index=False)
        return False

    # SMA
    for w in SMA_WINDOWS:
        df[f"SMA{w}"] = df["value"].rolling(window=w, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax2.bar(df["time"], df["volume"], width=0.8, color="gray", alpha=0.3, label="Volume")
    ax2.set_ylabel("Volume", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)

    ax1.plot(df["time"], df["value"], color="#ff99cc", lw=1.6, label="Index")
    colors = ["#80d0ff", "#ffd580", "#b0ffb0"]
    for i, w in enumerate(SMA_WINDOWS):
        ax1.plot(df["time"], df[f"SMA{w}"], lw=1.0, color=colors[i], label=f"SMA{w}")

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Value")
    fig.tight_layout()

    out_csv = f"docs/outputs/{key}_{label}.csv"
    out_png = f"docs/outputs/{key}_{label}.png"
    df[["time","value","volume"]].to_csv(out_csv, index=False)
    plt.legend(loc="upper left")
    plt.savefig(out_png, dpi=160)
    plt.close()
    log(f"saved: {out_csv}, {out_png}")
    return True

def main():
    key = os.environ.get("INDEX_KEY")
    if not key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    base = "docs/outputs"
    os.makedirs(base, exist_ok=True)

    src, kind = find_input(base, key)
    if not src:
        raise SystemExit(f"ERROR: input not found under {base}")

    raw = read_data(src, kind)
    daily = to_daily(raw, kind)

    now = datetime.now(tz=JST)
    ranges = {
        "7d": now - timedelta(days=7),
        "1m": now - timedelta(days=31),
        "1y": now - timedelta(days=365),
    }

    for label, since in ranges.items():
        sub = daily[daily["time"] >= since].copy()
        # データが極端に少ない場合も生成する
        if sub.empty and not daily.empty:
            sub = daily.tail(1).copy()
            log(f"insufficient data for {label}; using last point only")
        plot_chart(sub, key, label)

if __name__ == "__main__":
    main()
