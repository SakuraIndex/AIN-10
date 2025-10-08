#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== 共通スタイル（桜Indexダーク）=====
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
VOL_COL_CANDS = ["volume", "vol", "出来高"]

def log(m): print(f"[long_charts] {m}")

# ---------- 入力検索 ----------
def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    cand = find_first([
        os.path.join(base, f"{key}_intraday.csv"),
        os.path.join(base, f"{key}_intraday.txt"),
        os.path.join(base, f"{key}.csv"),
    ])
    if cand: return cand
    # 自動検出 fallback
    ii = sorted(glob.glob(os.path.join(base, "*_intraday.csv")) +
                glob.glob(os.path.join(base, "*_intraday.txt")))
    return ii[0] if ii else None

def find_history(base, key):
    cand = find_first([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand: return cand
    hh = sorted(glob.glob(os.path.join(base, "*_history.csv")) +
                glob.glob(os.path.join(base, "*_history.txt")))
    return hh[0] if hh else None

# ---------- 読み込み（柔軟な列検出） ----------
def parse_time_any(x):
    if pd.isna(x): return pd.NaT
    s = str(x)
    if re.fullmatch(r"\d{10}", s):  # epoch sec
        return datetime.fromtimestamp(int(s), tz=JST)
    try:
        t = pd.to_datetime(s)
        if t.tzinfo is None:
            t = t.tz_localize(JST)
        return t.tz_convert(JST)
    except Exception:
        return pd.NaT

def normalize_df(path):
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    tcols = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    vcols = [c for c in df.columns if any(k in c for k in ["close","price","value","index","終値","値"])]
    volcols = [c for c in df.columns if any(k in c for k in VOL_COL_CANDS)]

    tcol = tcols[0] if tcols else df.columns[0]
    vcol = vcols[0] if vcols else (df.columns[1] if len(df.columns)>1 else df.columns[0])
    volcol = volcols[0] if volcols else None

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(parse_time_any)
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

# ---------- 集約 ----------
def to_daily(df):
    if df.empty: return df.copy()
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(JST).dt.date
    g = (d.groupby("date", as_index=False)
          .agg({"value":"last","volume":"sum"}))
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(JST)
    return g[["time","value","volume"]].sort_values("time").reset_index(drop=True)

# ---------- プロット共通 ----------
def format_time_axis(ax, mode):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=JST))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=JST)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    if len(series)==0:
        ax.set_ylim(-1, 1)
        return
    ymin, ymax = float(pd.Series(series).min()), float(pd.Series(series).max())
    if ymin == ymax:
        pad = max(abs(ymin)*0.02, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        span = ymax - ymin
        pad = span * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

def plot_df(df, key, label, mode):
    out_csv = f"docs/outputs/{key}_{label}.csv"
    out_png = f"docs/outputs/{key}_{label}.png"
    df[["time","value","volume"]].to_csv(out_csv, index=False)

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.35)

    # 出来高（全ゼロなら非表示）
    has_volume = (df["volume"].fillna(0).abs().sum() > 0)
    ax2 = None
    if has_volume:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.28, label="Volume")
        ax2.set_ylabel("Volume")

    # 価格 + SMA
    if not df.empty:
        ax1.plot(df["time"], df["value"], color=COLOR_PRICE,
                 lw=1.8 if mode=="1d" else 1.6, label="Index")
        for i, w in enumerate(SMA_WINDOWS):
            s = df["value"].rolling(window=w, min_periods=1).mean()
            ax1.plot(df["time"], s, lw=1.1, color=COLOR_SMA[i], label=f"SMA{w}")

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Date"); ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode)
    apply_y_padding(ax1, df["value"] if not df.empty else [0])

    # 凡例結合
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    log(f"saved: {out_csv}, {out_png}")

# ---------- メイン ----------
def main():
    key = os.environ.get("INDEX_KEY")
    if not key: raise SystemExit("ERROR: INDEX_KEY not set")

    base = "docs/outputs"
    os.makedirs(base, exist_ok=True)

    intraday_path = find_intraday(base, key)
    history_path  = find_history(base, key)

    if intraday_path:
        log(f"input(intraday): {intraday_path}")
        intraday = normalize_df(intraday_path)
    else:
        intraday = pd.DataFrame(columns=["time","value","volume"])

    if history_path:
        log(f"input(history): {history_path}")
        history = normalize_df(history_path)
        history = to_daily(history)   # 念のため日次に整形
    else:
        history = pd.DataFrame(columns=["time","value","volume"])

    now = datetime.now(tz=JST)

    # ---- 1日：intraday をそのまま使用（無ければ history の最後の1日で代替）----
    since_1d = now - timedelta(days=1)
    df_1d = intraday[intraday["time"] >= since_1d].copy()
    if df_1d.empty:
        # 代替：日足の最後の1点だけでも描画
        last = history.tail(1).copy()
        if not last.empty:
            # ダミーで1日分の水平線にする
            t0 = now - timedelta(hours=6)
            t1 = now
            df_1d = pd.DataFrame({
                "time": [t0, t1],
                "value": [last["value"].iloc[0]]*2,
                "volume": [0,0]
            })
            log("1d: intraday empty; used last daily value as flat line")
    plot_df(df_1d, key, "1d", "1d")

    # ---- 1週間 / 1ヶ月 / 1年：history を優先、無ければ intraday を日足化 ----
    def daily_source():
        if not history.empty:
            return history
        return to_daily(intraday)

    daily = daily_source()

    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        since = now - timedelta(days=days)
        sub = daily[daily["time"] >= since].copy()
        if sub.empty and not daily.empty:
            sub = daily.tail(1).copy()
            log(f"{label}: no data; used last daily point")
        plot_df(sub, key, label, "long")

 

if __name__ == "__main__":
    main()
