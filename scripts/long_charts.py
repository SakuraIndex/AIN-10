#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 可変：指数ごとのローカル市場タイムゾーンと取引時間 ======
INDEX_KEY = os.environ.get("INDEX_KEY", "").strip()
if not INDEX_KEY:
    raise SystemExit("ERROR: INDEX_KEY not set")

def market_profile(index_key: str):
    if index_key.lower() == "ain10":
        return ("America/New_York", (9, 30), (16, 0))  # ET
    return ("Asia/Tokyo", (9, 0), (15, 0))             # デフォルト

TZ_NAME, SESSION_START, SESSION_END = market_profile(INDEX_KEY)

# Matplotlib 外観
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
COLOR_VOLUME = "#7f8ca6"

OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg: str):
    print(f"[long_charts] {msg}")

# ====== ファイル検出 ======
def _first_exists(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    cand = _first_exists([
        os.path.join(base, f"{key}_intraday.csv"),
        os.path.join(base, f"{key}_intraday.txt"),
    ])
    if cand: return cand
    gl = sorted(glob.glob(os.path.join(base, "*_intraday.csv")))
    return gl[0] if gl else None

def find_history(base, key):
    cand = _first_exists([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand: return cand
    gl = sorted(glob.glob(os.path.join(base, "*_history.csv")))
    return gl[0] if gl else None

# ====== 日時・列検出 ======
def parse_time_any(x, tz_name: str):
    if pd.isna(x): return pd.NaT
    s = str(x)
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(tz_name)
    try:
        t = pd.to_datetime(s, utc=False)
        if t.tzinfo is None:
            t = t.tz_localize(tz_name)
        else:
            t = t.tz_convert(tz_name)
        return t
    except Exception:
        return pd.NaT

def pick_value_column(df: pd.DataFrame) -> str:
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for k in ["close","price","value","index","終値","last","adjclose"]:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            return k
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) == 1:
        return numeric[0]
    if not numeric:
        for c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                return c
        raise KeyError("No numeric column for value")
    span = {c: float(pd.to_numeric(df[c], errors="coerce").dropna().max()
                     - pd.to_numeric(df[c], errors="coerce").dropna().min())
            for c in numeric}
    return max(span, key=span.get)

def pick_volume_column(df: pd.DataFrame):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for k in ["volume","vol","qty","quantity","count","turnover","amount",
              "出来高","売買高","出来高合計"]:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            return k
    return None

def normalize_df(path: str, tz_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    t_candidates = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = pick_value_column(df)
    volcol = pick_volume_column(df)

    out = pd.DataFrame()
    out["time"]   = df[tcol].apply(lambda x: parse_time_any(x, tz_name))
    out["value"]  = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time","value"])
    out = out.sort_values("time").reset_index(drop=True)
    log(f"read: {os.path.basename(path)} cols={raw_cols} -> time={tcol}, value={vcol}, volume={volcol or 'NONE'} rows={len(out)}")
    return out

def to_daily(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if df.empty: return df.copy()
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(tz_name).dt.date
    g = d.groupby("date", as_index=False).agg({"value":"last","volume":"sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(tz_name)
    return g[["time","value","volume"]].sort_values("time").reset_index(drop=True)

# ====== 軸・レンジ ======
def format_time_axis(ax, mode, tz_name):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz_name))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz_name))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz_name)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        ax.set_ylim(-1.0, 1.0); return
    ymin, ymax = float(s.min()), float(s.max())
    if ymin == ymax:
        pad = max(abs(ymin)*0.02, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        span = ymax - ymin
        pad = span * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

def session_bounds(day_ts: pd.Timestamp, tz_name: str, start_hm, end_hm):
    """day_ts は tz付き Timestamp（日付だけ使う）。"""
    start = pd.Timestamp(day_ts.year, day_ts.month, day_ts.day, start_hm[0], start_hm[1], tz=tz_name)
    end   = pd.Timestamp(day_ts.year, day_ts.month, day_ts.day, end_hm[0],   end_hm[1],   tz=tz_name)
    return start, end

# ====== ％→絶対値（必要時のみ） ======
def maybe_percent_to_absolute(df_1d: pd.DataFrame, base_value: float) -> pd.DataFrame:
    if df_1d.empty or base_value is None:
        return df_1d
    s = pd.to_numeric(df_1d["value"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return df_1d
    vmax = float(s.abs().max())
    if vmax <= 20.0 and base_value > 50:  # ％っぽい
        out = df_1d.copy()
        out["value"] = base_value * (1.0 + (out["value"] / 100.0))
        log(f"1d values looked like percentage; converted using base={base_value:.4f}")
        return out
    return df_1d

# ====== 描画 ======
def plot_df(df: pd.DataFrame, key: str, label: str, mode: str, tz_name: str, frame_limits=None):
    out_csv = os.path.join(OUTPUT_DIR, f"{key}_{label}.csv")
    out_png = os.path.join(OUTPUT_DIR, f"{key}_{label}.png")
    df[["time","value","volume"]].to_csv(out_csv, index=False)

    # 価格ラインの軽い補間
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
        now = pd.Timestamp.now(tz=tz_name)
        df_line = pd.DataFrame({"time":[now - pd.Timedelta(hours=1), now],
                                "value":[last_v, last_v]})

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.30)

    # 出来高（列が無い/全ゼロなら非表示）
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    has_vol = (df["volume"].abs().sum() > 0)
    ax2 = None
    if has_vol:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, label="Volume", zorder=1)
        ax2.set_ylabel("Volume")

    ax1.plot(df_line["time"], df_line["value"], color=COLOR_PRICE, lw=2.0 if mode=="1d" else 1.8,
             solid_capstyle="round", solid_joinstyle="round", antialiased=True, label="Index", zorder=3)

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Time" if mode=="1d" else "Date"); ax1.set_ylabel("Index Value")
    format_time_axis(ax1, mode if label=="1d" else "long", tz_name)
    apply_y_padding(ax1, df_line["value"])

    if frame_limits is not None:
        ax1.set_xlim(frame_limits[0], frame_limits[1])

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
    tz = TZ_NAME  # 例: "America/New_York" / "Asia/Tokyo"

    intraday_p = find_intraday(OUTPUT_DIR, INDEX_KEY)
    history_p  = find_history(OUTPUT_DIR, INDEX_KEY)

    intraday = normalize_df(intraday_p, tz) if intraday_p else pd.DataFrame(columns=["time","value","volume"])
    history  = normalize_df(history_p,  tz) if history_p  else pd.DataFrame(columns=["time","value","volume"])

    daily_all = to_daily(history if not history.empty else intraday, tz)

    # --- 1d：ローカルTZの最終日・米国市場は 09:30–16:00 に枠固定 ---
    frame = None
    if not intraday.empty:
        last_day = intraday["time"].dt.tz_convert(tz).dt.date.max()
        df_1d = intraday[intraday["time"].dt.tz_convert(tz).dt.date == last_day].copy()
        prev = daily_all[daily_all["time"].dt.date < last_day].tail(1)
        base_value = float(prev["value"].iloc[0]) if not prev.empty else None
        df_1d = maybe_percent_to_absolute(df_1d, base_value)

        # 枠を市場時間に合わせる
        day_ts = pd.Timestamp(last_day, tz=tz)
        start, end = session_bounds(day_ts, tz, SESSION_START, SESSION_END)
        frame = (start, end)
        log(f"1d frame: {start} ~ {end} ({TZ_NAME})")
    else:
        df_1d = pd.DataFrame(columns=["time","value","volume"])

    if df_1d.empty:
        last = daily_all.tail(1)
        if not last.empty:
            now = pd.Timestamp.now(tz=tz)
            df_1d = pd.DataFrame({"time":[now - pd.Timedelta(hours=6), now],
                                  "value":[float(last["value"].iloc[0])]*2,
                                  "volume":[0,0]})
            frame = None
            log("1d fallback: used last daily value")

    plot_df(df_1d, INDEX_KEY, "1d", "1d", tz, frame_limits=frame)

    # --- 7d/1m/1y ---
    now = pd.Timestamp.now(tz=tz)
    for label, days in [("7d",7), ("1m",31), ("1y",365)]:
        since = now - timedelta(days=days)
        sub = daily_all[daily_all["time"] >= since].copy()
        if sub.empty and not daily_all.empty:
            sub = daily_all.tail(1).copy()
            log(f"{label} fallback: last daily point only")
        plot_df(sub, INDEX_KEY, label, "long", tz)

if __name__ == "__main__":
    main()
