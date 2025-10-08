#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== 基本設定（ブランド寄せのダークテーマ）=====
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

# 線色（桜Indexっぽく）
COLOR_PRICE = "#ff99cc"
COLOR_SMA    = ["#80d0ff", "#ffd580", "#b0ffb0"]
COLOR_VOLUME = "#7f8ca6"

SMA_WINDOWS = [5, 25, 75]
VOLUME_COLUMN_CANDIDATES = ["volume", "vol", "出来高"]

def log(msg): print(f"[long_charts] {msg}")

# ---------- 入力検出 ----------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_input(base, key):
    """優先: intraday → history。無ければ自動検出"""
    cand = first_existing([
        os.path.join(base, f"{key}_intraday.csv"),
        os.path.join(base, f"{key}_intraday.txt"),
        os.path.join(base, f"{key}.csv"),
    ])
    if cand: return cand, "intraday"

    ii = sorted(glob.glob(os.path.join(base, "*_intraday.csv")) +
                glob.glob(os.path.join(base, "*_intraday.txt")))
    if ii: return ii[0], "intraday"

    cand = first_existing([
        os.path.join(base, f"{key}_history.csv"),
        os.path.join(base, f"{key}_history.txt"),
    ])
    if cand: return cand, "history"

    hh = sorted(glob.glob(os.path.join(base, "*_history.csv")) +
                glob.glob(os.path.join(base, "*_history.txt")))
    if hh: return hh[0], "history"

    return None, None

# ---------- 読み込み ----------
def read_data(path, kind):
    """任意CSV/TXTから (time, value, volume) に正規化"""
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    # ゆるい列名検出
    t_candidates = [c for c in df.columns if any(k in c for k in ["time","date","datetime","時刻","日付"])]
    v_candidates = [c for c in df.columns if any(k in c for k in ["close","price","value","index","終値","値"])]
    vol_candidates = [c for c in df.columns if any(k in c for k in VOLUME_COLUMN_CANDIDATES)]

    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = v_candidates[0] if v_candidates else (df.columns[1] if len(df.columns)>1 else df.columns[0])
    volcol = vol_candidates[0] if vol_candidates else None

    def parse_time(x):
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

    df["time"] = df[tcol].apply(parse_time)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    df["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    df = df.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return df[["time","value","volume"]]

def to_daily(df, kind):
    """intraday→日足(終値/出来高合計), history→ほぼそのまま"""
    if df.empty: return df.copy()
    if kind == "history":
        d = df.copy()
        d["date"] = d["time"].dt.tz_convert(JST).dt.date
        d = (d.groupby("date", as_index=False)
              .agg({"value":"last","volume":"sum"}))
        d["time"] = pd.to_datetime(d["date"]).dt.tz_localize(JST)
        return d[["time","value","volume"]].sort_values("time").reset_index(drop=True)

    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(JST).dt.date
    daily = (d.groupby("date", as_index=False)
              .agg({"value":"last","volume":"sum"}))
    daily["time"] = pd.to_datetime(daily["date"]).dt.tz_localize(JST)
    return daily[["time","value","volume"]].sort_values("time").reset_index(drop=True)

# ---------- 作図 ----------
def _format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    for label in ax.get_xticklabels():
        label.set_rotation(0)

def _apply_y_padding(ax, y):
    if len(y)==0: return
    ymin, ymax = float(pd.Series(y).min()), float(pd.Series(y).max())
    if ymin == ymax:
        pad = abs(ymin)*0.02 if ymin != 0 else 0.5
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        span = ymax - ymin
        pad = span * 0.08
        ax.set_ylim(ymin - pad, ymax + pad)

def plot_chart(df, key, label):
    out_csv = f"docs/outputs/{key}_{label}.csv"
    out_png = f"docs/outputs/{key}_{label}.png"

    # CSVは常に保存（デバッグにも便利）
    df[["time","value","volume"]].to_csv(out_csv, index=False)

    # データがゼロでも空白PNGは出す（サイト側の存在チェック用）
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.35)

    # 出来高（全ゼロなら非表示）
    has_volume = (df["volume"].fillna(0).abs().sum() > 0)
    if has_volume:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9, color=COLOR_VOLUME, alpha=0.28, label="Volume")
        ax2.set_ylabel("Volume")
        ax2.tick_params(axis="y")
    else:
        ax2 = None

    # 価格線
    if not df.empty:
        ax1.plot(df["time"], df["value"], color=COLOR_PRICE, lw=1.8, label="Index")
        for i, w in enumerate(SMA_WINDOWS):
            s = df["value"].rolling(window=w, min_periods=1).mean()
            ax1.plot(df["time"], s, lw=1.1, color=COLOR_SMA[i], label=f"SMA{w}")

    ax1.set_title(f"{key.upper()} ({label})", color="#ffb6c1", pad=10)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Value")

    _format_time_axis(ax1)
    _apply_y_padding(ax1, df["value"] if not df.empty else [0])

    # 凡例：両軸のハンドルを結合
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if has_volume else ([], []))
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

    src, kind = find_input(base, key)
    if not src:
        raise SystemExit(f"ERROR: input not found under {base}")
    log(f"input({kind}): {src}")

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
        # データが無い場合でも最後の一点で出す（サイト上の見栄え維持）
        if sub.empty and not daily.empty:
            sub = daily.tail(1).copy()
            log(f"insufficient data for {label}; using last point only")
        plot_chart(sub, key, label)

if __name__ == "__main__":
    main()
