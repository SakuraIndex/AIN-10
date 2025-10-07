# -*- coding: utf-8 -*-
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt

# ====== 設定 ======
INDEX_NAME = "AIN-10 Index (AI US)"
OUTPUT_TAG = "ain10"
# 公式ティッカー（CoreWeave=CRWV, Nebius=NBIS）
TICKERS: List[str] = [
    "MSFT", "GOOGL", "META", "AMZN", "PLTR", "ORCL", "NVDA", "AMD", "CRWV", "NBIS"
]
BASE_VALUE = 1000.0
OUT_DIR = "docs/outputs"
CSV, PNG, TXT = [
    os.path.join(OUT_DIR, f"{OUTPUT_TAG}_{x}")
    for x in ["history.csv", "chart.png", "post.txt"]
]
os.makedirs(OUT_DIR, exist_ok=True)
# ==================

def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def pick_close(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "Close" in df.columns:
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return s if len(s) else None
    if isinstance(df.columns, pd.MultiIndex):
        for c in df.columns:
            if isinstance(c, tuple) and c[-1] == "Close":
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s):
                    return s
    return None

def fetch_series(t, start="2024-01-01") -> Optional[pd.Series]:
    df = yf.download(
        t,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )
    return pick_close(df)

def fetch_mcap(t) -> Optional[float]:
    try:
        fi = yf.Ticker(t).fast_info
        mc = getattr(fi, "market_cap", None)
        if mc is None:
            mc = yf.Ticker(t).info.get("marketCap")
        return float(mc) if mc else None
    except Exception:
        return None

def build_cap_idx(series_map: Dict[str, pd.Series], cols: List[str]) -> pd.Series:
    df = pd.concat([series_map[c] for c in cols if c in series_map], axis=1)
    df.columns = [c for c in cols if c in series_map]
    df = df.sort_index().ffill().dropna(how="all")
    ret = df.pct_change().dropna(how="all")

    weights = []
    for c in df.columns:
        mc = fetch_mcap(c)
        weights.append(mc if mc and mc > 0 else 0.0)
    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()

    basket = (ret * w).sum(axis=1)
    idx = pd.Series(BASE_VALUE, index=[ret.index[0] - pd.Timedelta(days=1)])
    return pd.concat([idx, BASE_VALUE * (1 + basket).cumprod()])

def main():
    print(f"[INFO] Building {INDEX_NAME} ...")
    series_map = {}
    for t in TICKERS:
        s = fetch_series(t)
        if s is not None and len(s):
            series_map[t] = s
        else:
            print(f"[WARN] skip: {t}")
    if not series_map:
        raise RuntimeError("No data fetched.")

    idx = build_cap_idx(series_map, TICKERS)

    # 既存履歴とマージ
    if os.path.exists(CSV):
        old = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")["index_value"]
        merged = pd.concat([old, idx], axis=1)
        merged.columns = ["old", "new"]
        final = merged["new"].fillna(merged["old"])
    else:
        final = idx
    final = final[~final.index.duplicated(keep="last")].sort_index()

    pd.DataFrame({"date": final.index, "index_value": final.values}).to_csv(
        CSV, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] CSV saved: {CSV}")

    # 変化率
    chg = 0.0 if len(final) < 2 else (final.iloc[-1] / final.iloc[-2] - 1) * 100

    # チャート描画
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(final.index, final.values, linewidth=2.5)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{INDEX_NAME} (Base={BASE_VALUE:.0f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index")
    fig.tight_layout()
    plt.savefig(PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] PNG saved: {PNG}")

    # 投稿文
    sign = "＋" if chg >= 0 else "－"
    with open(TXT, "w", encoding="utf-8") as f:
        f.write(
            f"{INDEX_NAME} 日次スナップショット（{jst_now():%Y/%m/%d}）\n"
            f"{sign}{abs(chg):.2f}%（前日比） / 基準{BASE_VALUE:.0f}\n"
            f"構成銘柄: {', '.join([c for c in TICKERS if c in series_map])}\n"
            f"#桜Index #AIN10\n"
        )
    print(f"[OK] TXT saved: {TXT}")

if __name__ == "__main__":
    main()
