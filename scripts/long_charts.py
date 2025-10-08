#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

JST = timezone(timedelta(hours=9))

# ====== 見た目（桜Indexダーク）======
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

def log(m): print(f"[long_charts] {m}")

OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# INDEX_KEY はワークフローの env で渡す（例：ain10 / astra4 / scoin_plus / rbank9）
INDEX_KEY = os.environ.get("INDEX_KEY", "").strip()
if not INDEX_KEY:
    raise SystemExit("ERROR: INDEX_KEY not set")

# ---------- 入力ファイル検出 ----------
def find_first(patterns):
    for p in patterns:
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
    gl = sorted(glob.glob(os.path.join(base, "*_intraday.csv")) +
                glob.glob(os.path.join(base, "*_intraday.txt")))
    return gl[0] if gl else None

def find_history(base, key):
    cand = find_first([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand: return cand
    gl = sorted(glob.glob(os.path.join(base, "*_history.csv")) +
                glob.glob(os.path.join(base, "*_history.txt")))
    return gl[0] if gl else None

# ---------- ユーティリティ ----------
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

def pick_value_column(df):
    """
    値段列の自動検出:
    - キーワード優先
    - 数値列が1本だけならそれ
    - そうでなければ、値の変動が最も大きい数値列
    """
    cols = [c.lower().strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    # 1) キーワード優先
    keys = ["close","price","value","index","終値","終値(円)","last","adjclose",
            INDEX_KEY, INDEX_KEY.replace("_",""), INDEX_KEY.split("_")[0]]
    for k in keys:
        if k in df.columns:
            if pd.api.types.is_numeric_dtype(df[k]):
                return k

    # 2) 数値列抽出
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    if not numeric_cols:
        # 文字列しかない場合、最初の列を数値化トライ
        for c in df.columns:
            try_col = pd.to_numeric(df[c], errors="coerce")
            if try_col.notna().sum() > 0:
                df[c] = try_col
                return c
        # 何も無理ならエラー
        raise KeyError("No numeric column found for value")

    # 3) 変動が最も大きい列を選ぶ
    vol = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            vol[c] = -math.inf
        else:
            vol[c] = float(s.max() - s.min())
    # 最大変動列
    best = max(vol, key=vol.get)
    return best

def normalize_df(path):
    df = pd.read_csv(path)
    raw_cols = df.columns.tolist()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # 時刻列
    t_candidates = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    tcol = t_candidates[0] if t_candidates else df.columns[0]

    # 値列（自動検出）
    vcol = pick_value_column(df)

    # 出来高
    vol_candidates = [c for c in df.columns if any(k in c for k in ["volume","vol","出来高"])]
    volcol = vol_candidates[0] if vol_candidates else None

    out = pd.DataFrame()
    out["time"]   = df[tcol].apply(parse_time_any)
    out["value"]  = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

    log(f"read: {os.path.basename(path)}  cols(raw)={raw_cols} -> time='{tcol}', value='{vcol}', volume='{volcol or 'NONE'}'  rows={len(out)}")
    return out

def to_daily(df):
    if df.empty: return df.copy()
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(JST).dt.date
    g = (d.groupby("date", as_index=False)
          .agg({"value":"last","volume":"sum"}))
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(JST)
    return g[["time","value","volume"]].sort_values("time").reset_index(drop=True)

# ---------- 描画 ----------
def format_time_axis(ax, mode):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=JST))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=JST))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=JST)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    if len(series)==0:
        ax.set_ylim(-1,1); return
    s = pd.Series(series).astype(float)
    ymin, ymax = s.min(), s.max()
    if ymin == ymax:
        pad = max(abs(ymin)*0.02, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        span = ymax - ymin
        pad = span * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

def plot_df(df, key, label, mode):
    out_csv = os.path.join(OUTPUT_DIR, f"{key}_{label}.csv")
    out_png = os.path.join(OUTPUT_DIR, f"{key}_{label}.png")
    df[["time","value","volume"]].to_csv(out_csv, index=False)

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.35)

    # volume
    has_vol = (df["volume"].fillna(0).abs().sum() > 0)
    ax2 = None
    if has_vol:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.28, label="Volume")
        ax2.set_ylabel("Volume")

    # price + SMA
    if not df.empty:
        ax1.plot(df["time"], df["value"], color=COLOR_PRICE, lw=1.8 if mode=="1d" else 1.6, label="Index")
        for i, w in enumerate(SMA_WINDOWS):
            s = df["value"].rolling(window=w, min_periods=1).mean()
            ax1.plot(df["time"], s, lw=1.1, color=COLOR_SMA[i], label=f"SMA{w}")

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Date"); ax1.set_ylabel("Index Value")
    format_time_axis(ax1, mode)
    apply_y_padding(ax1, df["value"] if not df.empty else [0])

    # legend merge
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = (ax2.get_legend_handles_labels() if ax2 else ([],[]))
    if h1 or h2:
        ax1.legend(h1+h2, l1+l2, loc="upper left", frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    log(f"saved: {os.path.basename(out_png)}")

# ---------- メイン ----------
def main():
    base = OUTPUT_DIR

    intraday_p = find_intraday(base, INDEX_KEY)
    history_p  = find_history(base, INDEX_KEY)

    intraday = normalize_df(intraday_p) if intraday_p else pd.DataFrame(columns=["time","value","volume"])
    history  = normalize_df(history_p)  if history_p  else pd.DataFrame(columns=["time","value","volume"])

    # 1d: intraday をそのまま。なければ daily の最後からフラット線
    now = datetime.now(tz=JST)
    since_1d = now - timedelta(days=1)
    df_1d = intraday[intraday["time"] >= since_1d].copy()
    if df_1d.empty:
        daily = to_daily(history if not history.empty else intraday)
        last = daily.tail(1)
        if not last.empty:
            t0 = now - timedelta(hours=6)
            t1 = now
            df_1d = pd.DataFrame({
                "time":[t0,t1],
                "value":[float(last["value"].iloc[0])]*2,
                "volume":[0,0],
            })
            log("1d fallback: used last daily value as flat line")
    plot_df(df_1d, INDEX_KEY, "1d", "1d")

    # long: history 優先、無ければ intraday を日足化
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
