#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) with timezone handling and 1d bull/bear coloring.
- Intraday CSV is assumed in ET for AIN10, converted to JST for display.
- History CSV is assumed in JST.
- 1d line color: green if close > open, red if close < open, gray if equal.
"""

import os
import re
import glob
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- constants ----------
OUTPUT_DIR = "docs/outputs"

# UI colors
COLOR_PRICE_DEFAULT = "#ff99cc"
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線（青緑）
COLOR_DOWN = "#FF4C4C" # 陰線（赤）
COLOR_EQUAL = "#CCCCCC"

# Matplotlib style
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "figure.facecolor": "#0b0f1a",
    "axes.facecolor":   "#0b0f1a",
    "axes.edgecolor":   "#27314a",
    "axes.labelcolor":  "#e5ecff",
    "xtick.color":      "#b8c2e0",
    "ytick.color":      "#b8c2e0",
    "grid.color":       "#27314a",
})

def log(msg: str):
    print(f"[long_charts] {msg}")

# ---------- index profile ----------
def market_profile(index_key: str):
    k = (index_key or "").lower()
    if k == "ain10":
        # AIN-10: intradayはET、historyはJST、表示はJST
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )
    # default（必要なら拡張）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

INDEX_KEY = os.environ.get("INDEX_KEY", "").strip()
if not INDEX_KEY:
    raise SystemExit("ERROR: INDEX_KEY not set")

MP = market_profile(INDEX_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- file finders ----------
def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    p = _first([f"{base}/{key}_intraday.csv", f"{base}/{key}_intraday.txt"])
    if p:
        return p
    found = sorted(glob.glob(f"{base}/*_intraday.csv"))
    return found[0] if found else None

def find_history(base, key):
    p = _first([f"{base}/{key}_history.csv", f"{base}/{key}_history.txt"])
    if p:
        return p
    found = sorted(glob.glob(f"{base}/*_history.csv"))
    return found[0] if found else None

# ---------- parsing ----------
def parse_time_any(x, raw_tz: str, display_tz: str):
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    if re.fullmatch(r"\d{10}", s):
        # epoch seconds -> assume UTC, then to display_tz
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    try:
        t = pd.to_datetime(s, utc=False)
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        else:
            t = t.tz_convert(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

def pick_value_col(df: pd.DataFrame) -> str:
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for k in ["close", "price", "value", "index", "終値", "last", "adjclose"]:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            return k
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) == 1:
        return nums[0]
    if not nums:
        for c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                return c
        raise KeyError("numeric value column not found")
    span = {}
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            span[c] = -np.inf
        else:
            span[c] = float(s.max() - s.min())
    return max(span, key=span.get)

def pick_volume_col(df: pd.DataFrame):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for k in ["volume", "vol", "qty", "quantity", "count", "turnover", "amount", "出来高", "売買高", "出来高合計"]:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            return k
    return None

def read_any(path: str, raw_tz: str, display_tz: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = pd.read_csv(path)
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]
    t_candidates = [c for c in df.columns if any(k in c for k in ["time", "date", "datetime", "時刻", "日付"])]
    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "value"])
    out = out.sort_values("time").reset_index(drop=True)
    log(f"read: {os.path.basename(path)} cols={raw_cols} -> time={tcol}, value={vcol}, volume={volcol or 'NONE'} rows={len(out)}")
    return out

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]].sort_values("time").reset_index(drop=True)

# ---------- axes helpers ----------
def format_time_axis(ax, mode, tz):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        ax.set_ylim(-1, 1)
        return
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        pad = max(abs(lo) * 0.02, 0.5)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)

def et_session_to_jst_frame(base_ts_jst: pd.Timestamp, session_tz: str, display_tz: str, start_hm, end_hm):
    et = base_ts_jst.tz_convert(session_tz)
    et_date = et.date()
    start_et = pd.Timestamp(et_date.year, et_date.month, et_date.day, start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_date.year, et_date.month, et_date.day, end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

# ---------- plotting ----------
def plot_df(df: pd.DataFrame, key: str, label: str, mode: str, tz: str, frame=None, resample_for_1d=False):
    # dump CSV for debug
    df[["time", "value", "volume"]].to_csv(f"{OUTPUT_DIR}/{key}_{label}.csv", index=False)

    # 1d は陽線/陰線で色分け
    if mode == "1d" and not df.empty:
        open_price = float(df["value"].iloc[0])
        close_price = float(df["value"].iloc[-1])
        if close_price > open_price:
            color_line = COLOR_UP
        elif close_price < open_price:
            color_line = COLOR_DOWN
        else:
            color_line = COLOR_EQUAL
    else:
        color_line = COLOR_PRICE_DEFAULT

    # データ整形：1d はほぼそのまま（軽い平滑）、長期はリサンプル
    if mode == "1d" and not resample_for_1d:
        line = df.copy()
        if not line.empty:
            line["value"] = line["value"].rolling(3, center=True, min_periods=1).mean()
    else:
        if not df.empty:
            ts = df.set_index("time").sort_index()
            freq = {"1d": "15T", "7d": "6H", "1m": "1D", "1y": "3D"}.get(label, "1D")
            line = ts[["value"]].resample(freq).interpolate("time").ffill().bfill().reset_index()
            line["value"] = line["value"].rolling(3, center=True, min_periods=1).mean()
        else:
            line = df.copy()

    if not line.empty:
        line["value"] = pd.to_numeric(line["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        line = line.dropna(subset=["value"]).reset_index(drop=True)

    if line.empty:
        now = pd.Timestamp.now(tz=tz)
        last_v = float(df["value"].tail(1).values[0]) if not df.empty else 0.0
        line = pd.DataFrame({"time": [now - pd.Timedelta(hours=1), now], "value": [last_v, last_v]})

    # plot
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.30)

    # volume（あれば）
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")
        ax2.set_ylabel("Volume")

    ax1.plot(
        line["time"],
        line["value"],
        color=color_line,
        lw=2.2 if mode == "1d" else 1.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        label="Index",
        zorder=3,
    )

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")
    format_time_axis(ax1, mode if label == "1d" else "long", tz)
    apply_y_padding(ax1, line["value"])
    if frame is not None:
        ax1.set_xlim(frame[0], frame[1])

    leg = ax1.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color("#e5ecff")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{key}_{label}.png", dpi=180)
    plt.close()
    log(f"saved: {key}_{label}.png (color={color_line})")

# ---------- main ----------
def main():
    RAW_TZ_INTRADAY = MP["RAW_TZ_INTRADAY"]
    RAW_TZ_HISTORY = MP["RAW_TZ_HISTORY"]
    DISPLAY_TZ = MP["DISPLAY_TZ"]
    SESSION_TZ = MP["SESSION_TZ"]
    START_HM = MP["SESSION_START"]
    END_HM = MP["SESSION_END"]

    intraday_p = find_intraday(OUTPUT_DIR, INDEX_KEY)
    history_p = find_history(OUTPUT_DIR, INDEX_KEY)

    intraday = read_any(intraday_p, RAW_TZ_INTRADAY, DISPLAY_TZ) if intraday_p else pd.DataFrame(columns=["time", "value", "volume"])
    history = read_any(history_p, RAW_TZ_HISTORY, DISPLAY_TZ) if history_p else pd.DataFrame(columns=["time", "value", "volume"])

    daily_all = to_daily(history if not history.empty else intraday, DISPLAY_TZ)

    # 1d frame（ET 09:30–16:00 -> JST）
    if not intraday.empty:
        last_ts = intraday["time"].max()  # already JST
        start_jst, end_jst = et_session_to_jst_frame(last_ts, SESSION_TZ, DISPLAY_TZ, START_HM, END_HM)
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        # %っぽい場合は前日終値で絶対値化
        prev = daily_all[daily_all["time"] < start_jst].tail(1)
        base = float(prev["value"].iloc[0]) if not prev.empty else None
        if not df_1d.empty:
            rng = pd.to_numeric(df_1d["value"], errors="coerce").abs()
            if rng.max() <= 20.0 and base and base > 50:
                df_1d["value"] = base * (1.0 + df_1d["value"] / 100.0)
                log(f"converted 1d % -> absolute (base={base:.4f})")
        frame_1d = (start_jst, end_jst)
        log(f"1d frame JST: {start_jst} ~ {end_jst} rows={len(df_1d)}")
    else:
        df_1d = pd.DataFrame(columns=["time", "value", "volume"])
        frame_1d = None

    if df_1d.empty:
        last = daily_all.tail(1)
        if not last.empty:
            now = pd.Timestamp.now(tz=DISPLAY_TZ)
            df_1d = pd.DataFrame({"time": [now - pd.Timedelta(hours=1), now],
                                  "value": [float(last['value'].iloc[0])] * 2,
                                  "volume": [0, 0]})
            frame_1d = None
            log("1d fallback: last daily value")

    plot_df(df_1d, INDEX_KEY, "1d", "1d", DISPLAY_TZ, frame=frame_1d, resample_for_1d=False)

    # 7d / 1m / 1y
    now = pd.Timestamp.now(tz=DISPLAY_TZ)
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))].copy()
        if sub.empty and not daily_all.empty:
            sub = daily_all.tail(1).copy()
            log(f"{label} fallback: last daily point only")
        plot_df(sub, INDEX_KEY, label, "long", DISPLAY_TZ)

if __name__ == "__main__":
    main()
