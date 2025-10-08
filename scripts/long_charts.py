#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 共通設定 ======
JST = timezone(timedelta(hours=9))
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

COLOR_PRICE  = "#ff99cc"
COLOR_SMA    = ["#80d0ff", "#ffd580", "#b0ffb0"]
COLOR_VOLUME = "#7f8ca6"
SMA_WINDOWS = [5, 25, 75]
OUTPUT_DIR = "docs/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
INDEX_KEY = os.environ.get("INDEX_KEY", "").strip()
if not INDEX_KEY:
    raise SystemExit("ERROR: INDEX_KEY not set")

def log(msg):
    print(f"[long_charts] {msg}")

# ====== ファイル検出 ======
def find_first(patterns):
    for p in patterns:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    cand = find_first([
        os.path.join(base, f"{key}_intraday.csv"),
        os.path.join(base, f"{key}_intraday.txt"),
    ])
    if cand: return cand
    gl = sorted(glob.glob(os.path.join(base, "*_intraday.csv")))
    return gl[0] if gl else None

def find_history(base, key):
    cand = find_first([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand: return cand
    gl = sorted(glob.glob(os.path.join(base, "*_history.csv")))
    return gl[0] if gl else None

# ====== データ整形 ======
def parse_time_any(x):
    if pd.isna(x): return pd.NaT
    s = str(x)
    if re.fullmatch(r"\d{10}", s):
        return datetime.fromtimestamp(int(s), tz=JST)
    try:
        t = pd.to_datetime(s)
        if t.tzinfo is None:
            t = t.tz_localize(JST)
        return t.tz_convert(JST)
    except Exception:
        return pd.NaT

def pick_value_column(df):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    keys = ["close","price","value","index","終値","last","adjclose",
            INDEX_KEY, INDEX_KEY.replace("_",""), INDEX_KEY.split("_")[0]]
    for k in keys:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            return k
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return num_cols[0]
    if not num_cols:
        for c in df.columns:
            try_col = pd.to_numeric(df[c], errors="coerce")
            if try_col.notna().sum() > 0:
                return c
        raise KeyError("No numeric column found for value")
    vol = {c: float(df[c].max() - df[c].min()) for c in num_cols}
    return max(vol, key=vol.get)

def normalize_df(path):
    df = pd.read_csv(path)
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    t_candidates = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = pick_value_column(df)
    vol_candidates = [c for c in df.columns if any(k in c for k in ["volume","vol","出来高"])]
    volcol = vol_candidates[0] if vol_candidates else None

    out = pd.DataFrame()
    out["time"]   = df[tcol].apply(parse_time_any)
    out["value"]  = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time","value"])
    out = out.sort_values("time").reset_index(drop=True)
    log(f"read: {os.path.basename(path)} cols={raw_cols} -> time={tcol}, value={vcol}, volume={volcol or 'NONE'} rows={len(out)}")
    return out

def to_daily(df):
    if df.empty: return df.copy()
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(JST).dt.date
    g = d.groupby("date", as_index=False).agg({"value":"last","volume":"sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(JST)
    return g[["time","value","volume"]].sort_values("time").reset_index(drop=True)

# ====== グラフ整形 ======
def format_time_axis(ax, mode):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=JST))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=JST))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=JST)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        ax.set_ylim(-1.0, 1.0)
        return
    ymin, ymax = float(s.min()), float(s.max())
    if ymin == ymax:
        pad = max(abs(ymin)*0.02, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        span = ymax - ymin
        pad = span * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

# ====== 描画 ======
def plot_df(df, key, label, mode):
    out_csv = os.path.join(OUTPUT_DIR, f"{key}_{label}.csv")
    out_png = os.path.join(OUTPUT_DIR, f"{key}_{label}.png")
    df[["time","value","volume"]].to_csv(out_csv, index=False)

    # --- 補間 ---
    if not df.empty:
        ts = df.set_index("time").sort_index()
        freq = {"1d":"15T", "7d":"6H", "1m":"1D", "1y":"3D"}.get(label, "1D")
        ts_i = ts[["value"]].resample(freq).interpolate("time").ffill().bfill()
        df_line = ts_i.reset_index().rename(columns={"index":"time"})
        df_line["value"] = df_line["value"].rolling(window=3, center=True, min_periods=1).mean()
    else:
        df_line = df.copy()

    if not df_line.empty:
        df_line["value"] = pd.to_numeric(df_line["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df_line = df_line.dropna(subset=["value"]).reset_index(drop=True)
    if df_line.empty:
        base = df.dropna(subset=["value"]).tail(1)
        last_v = float(base["value"].values[0]) if not base.empty else 0.0
        now = pd.Timestamp.now(tz=JST)
        df_line = pd.DataFrame({
            "time": [now - pd.Timedelta(hours=1), now],
            "value": [last_v, last_v]
        })

    # --- グラフ ---
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.30)

    # volume処理
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    has_vol = (df["volume"].abs().sum() > 0)
    ax2 = None
    if has_vol:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.25, label="Volume", zorder=1)
        ax2.set_ylabel("Volume")

    # price + SMA
    lw_main = 2.0 if mode=="1d" else 1.8
    ax1.plot(df_line["time"], df_line["value"], color=COLOR_PRICE, lw=lw_main,
             solid_capstyle="round", solid_joinstyle="round", antialiased=True,
             label="Index", zorder=3)

    for i, w in enumerate(SMA_WINDOWS):
        s = df.set_index("time")["value"].rolling(window=w, min_periods=1).mean()
        s_i = s.resample(freq).interpolate("time").ffill().bfill()
        s_i = s_i.rolling(window=3, center=True, min_periods=1).mean()
        ax1.plot(s_i.index, s_i.values, lw=1.2, color=COLOR_SMA[i],
                 solid_capstyle="round", solid_joinstyle="round",
                 antialiased=True, label=f"SMA{w}", zorder=2)

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Date"); ax1.set_ylabel("Index Value")
    format_time_axis(ax1, mode if label=="1d" else "long")
    apply_y_padding(ax1, df_line["value"])

    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = (ax2.get_legend_handles_labels() if ax2 else ([],[]))
    if h1 or h2:
        leg = ax1.legend(h1+h2, l1+l2, loc="upper left", frameon=False)
        for t in leg.get_texts(): t.set_color("#e5ecff")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    log(f"saved: {os.path.basename(out_png)}")

# ====== メイン ======
def main():
    base = OUTPUT_DIR
    intraday_p = find_intraday(base, INDEX_KEY)
    history_p  = find_history(base, INDEX_KEY)
    intraday = normalize_df(intraday_p) if intraday_p else pd.DataFrame(columns=["time","value","volume"])
    history  = normalize_df(history_p)  if history_p  else pd.DataFrame(columns=["time","value","volume"])

    now = datetime.now(tz=JST)
    since_1d = now - timedelta(days=1)
    df_1d = intraday[intraday["time"] >= since_1d].copy()
    if df_1d.empty:
        daily = to_daily(history if not history.empty else intraday)
        last = daily.tail(1)
        if not last.empty:
            t0, t1 = now - timedelta(hours=6), now
            df_1d = pd.DataFrame({
                "time":[t0,t1],
                "value":[float(last["value"].iloc[0])]*2,
                "volume":[0,0],
            })
            log("1d fallback: used last daily value")
    plot_df(df_1d, INDEX_KEY, "1d", "1d")

    daily = to_daily(history if not history.empty else intraday)
    for label, days in [("7d",7), ("1m",31), ("1y",365)]:
        since = now - timedelta(days=days)
        sub = daily[daily["time"] >= since].copy()
        if sub.empty and not daily.empty:
            sub = daily.tail(1).copy()
            log(f"{label} fallback: last daily point only")
        plot_df(sub, INDEX_KEY, label, "long")

if __name__ == "__main__":
    main()
