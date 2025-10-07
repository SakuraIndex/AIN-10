# -*- coding: utf-8 -*-
"""
AIN-10 (AI US) Intraday Snapshot
- 米国AIセクター10社の1日（当日）指数チャート
- 5分足で算出
- 出力: docs/outputs/ain10_intraday.csv / ain10_intraday.png / ain10_intraday_post.txt
"""

import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ====== 設定 ======
TICKERS = [
    "ORCL", "PLTR", "GOOGL", "MSFT", "CRWV", "META",
    "NVDA", "AMD", "NBIS", "AMZN"
]

OUT_DIR = "docs/outputs"
CSV_PATH = os.path.join(OUT_DIR, "ain10_intraday.csv")
IMG_PATH = os.path.join(OUT_DIR, "ain10_intraday.png")
TXT_PATH = os.path.join(OUT_DIR, "ain10_intraday_post.txt")

INTERVAL = "5m"  # 5分足
PERIOD = "1d"    # 当日1日分
# ===================


def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def fetch_intraday(ticker: str) -> pd.Series:
    """ティッカーごとに当日5分足を取得して終値Seriesを返す"""
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            prepost=False,
            threads=False,
        )
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        s.index = s.index.tz_localize(None)
        return s
    except Exception as e:
        print(f"[WARN] fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def build_index(price_df: pd.DataFrame) -> pd.Series:
    """等加重のAIN-10インデックスを計算"""
    returns = price_df.pct_change()
    basket = returns.mean(axis=1, skipna=True)
    index_series = (1 + basket).cumprod() * 100
    index_series.name = "AIN-10"
    return index_series


def plot_chart(index_series: pd.Series):
    """チャート描画：陽線は青緑、陰線は赤"""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 当日の変化で色を分ける
    color = "#00ffff" if index_series.iloc[-1] >= index_series.iloc[0] else "#ff5050"

    ax.plot(index_series.index, index_series.values, color=color, linewidth=3, label="AIN-10")
    ax.legend(facecolor="black", labelcolor="white")

    ax.set_title(f"AIN-10 Intraday Snapshot ({jst_now():%Y/%m/%d %H:%M})", color="white", fontsize=18)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Open (%)", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#444")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart saved: {IMG_PATH}")


def main():
    print("[INFO] Fetching intraday data...")
    ensure_outdir()

    series_map = {}
    for t in TICKERS:
        s = fetch_intraday(t)
        if not s.empty:
            series_map[t] = s
        else:
            print(f"[WARN] No data for {t}")

    if not series_map:
        raise RuntimeError("No intraday data for any ticker.")

    df = pd.concat(series_map, axis=1)
    df = df.ffill().dropna(how="all")

    # リターン計算
    base = df.iloc[0]
    df_change = (df / base - 1) * 100
    index_series = df_change.mean(axis=1, skipna=True)

    # 保存
    pd.DataFrame(index_series).to_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"[OK] CSV saved: {CSV_PATH}")

    # プロット
    plot_chart(index_series)

    # 投稿文
    chg = index_series.iloc[-1]
    sign = "🔺" if chg >= 0 else "🔻"
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} AIN-10日中取引（{jst_now():%Y/%m/%d %H:%M}）\n"
            f"{chg:+.2f}%（当日始値比）\n"
            f"構成銘柄：{ ' / '.join(TICKERS) }\n"
            f"#AI株 #AIN10 #米国株\n"
        )

    print(f"[OK] TXT saved: {TXT_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()
