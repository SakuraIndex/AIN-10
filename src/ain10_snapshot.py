# -*- coding: utf-8 -*-
"""
AIN-10 (AI US) Daily Snapshot
- 10銘柄等金額平均の指数を日次で計算
- どの列構造でも「終値」を堅牢に取得
- 欠損日は自動スキップ、利用可能銘柄の平均で算出
- 出力: docs/outputs/ain10_history.csv, ain10_chart.png, ain10_post.txt
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== 設定 =====
TICKERS = [
    "ORCL",    # Oracle
    "PLTR",    # Palantir
    "GOOGL",   # Alphabet A
    "MSFT",    # Microsoft
    "CRWV",    # CoreWeave (recent IPO)
    "META",    # Meta
    "NVDA",    # NVIDIA
    "AMD",     # AMD
    "NBIS",    # Nebius (recent IPO)
    "AMZN",    # Amazon
]

OUT_DIR = "docs/outputs"
HIS_CSV = os.path.join(OUT_DIR, "ain10_history.csv")
IMG_PATH = os.path.join(OUT_DIR, "ain10_chart.png")
POST_PATH = os.path.join(OUT_DIR, "ain10_post.txt")

BASE_DATE = "2024-01-01"
BASE_VALUE = 100.0


# ===== ユーティリティ =====
def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def pick_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance.download の返す df から「終値」を確実に Series で取り出す。
    - 単一列: ['Close'] / ['Adj Close']
    - MultiIndex: ('Close', ticker) / (ticker, 'Close')
    - どれも無い場合は空Seriesを返す
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    s = None

    # まず MultiIndex かどうか
    if isinstance(df.columns, pd.MultiIndex):
        # 代表的な2パターンに対応
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif (ticker, "Close") in df.columns:
            s = df[(ticker, "Close")]
        elif (ticker, "Adj Close") in df.columns:
            s = df[(ticker, "Adj Close")]
    else:
        # 単層
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]

    if s is None:
        # どうしても取れない場合は空
        return pd.Series(dtype=float)

    # 数値化してNA除去
    s = pd.to_numeric(s, errors="coerce").dropna()
    # タイムゾーン除去（後続処理のため）
    if hasattr(s.index, "tz_localize"):
        try:
            s.index = s.index.tz_localize(None)
        except Exception:
            pass

    return s


def fetch_series(ticker: str) -> pd.Series:
    """
    単一銘柄の終値 Series を取得。
    返り値は timezone無しIndexの float Series。
    """
    # group_by は環境差異で列構造が変わるため、どちらでも取れるよう pick_close で吸収する
    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        # group_by="column" でも "ticker" でもOK（pick_closeが面倒見ます）
    )
    return pick_close(df, ticker)


def build_index(price_df: pd.DataFrame) -> pd.Series:
    """
    価格DataFrame（列=ティッカー, 行=日付, 値=終値）から
    等金額平均の指数（BASE_DATE=100）を作成。
    """
    price_df = price_df.sort_index()
    # リターン（等加重）：行ごとに利用可能な銘柄の平均リターンを使う
    rets = price_df.pct_change()
    mean_rets = rets.mean(axis=1, skipna=True)

    # BASE_DATE 以降で指数化
    base = pd.Timestamp(BASE_DATE)
    mean_rets = mean_rets[mean_rets.index >= base]
    index_series = (1.0 + mean_rets).cumprod() * BASE_VALUE
    index_series.name = "AIN-10"

    return index_series


def plot_index(index_series: pd.Series):
    # チャート色は陽線/陰線ではなく固定（見やすさ優先）
    line_color = "#34d1bf"  # ティール系

    plt.close("all")
    fig = plt.figure(figsize=(18, 9), dpi=140)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(index_series.index, index_series.values, color=line_color, linewidth=3.0, label="AIN-10")
    ax.legend(loc="best")

    # 軸・罫線のスタイル
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_title(f"AIN-10 (AI US) Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})", color="white", fontsize=22, pad=14)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Index (Base=100)", color="white")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    print("[INFO] Building AIN-10 Index (AI US) ...")
    ensure_outdir()

    series_dict = {}
    for t in TICKERS:
        try:
            print(f"[INFO] Fetching {t} ...")
            s = fetch_series(t)
            if not s.empty:
                series_dict[t] = s
            else:
                print(f"[WARN] empty series for {t}")
        except Exception as e:
            print(f"[WARN] fetch failed for {t}: {e}")

    if len(series_dict) == 0:
        raise RuntimeError("no series fetched for any ticker.")

    prices = pd.concat(series_dict, axis=1)  # 列=ティッカー
    index_series = build_index(prices)

    # 出力（CSV）
    df_out = pd.DataFrame({"AIN10": index_series})
    df_out.to_csv(HIS_CSV, encoding="utf-8")

    # チャート
    plot_index(index_series)

    # 前日比（直近2点から算出）
    change_pct = 0.0
    if len(index_series) >= 2:
        change_pct = (index_series.iloc[-1] / index_series.iloc[-2] - 1.0) * 100.0

    # 投稿文
    sign = "🔺" if change_pct >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} AIN-10 日次スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{change_pct:+.2f}%（前日終値比）\n"
            f"構成銘柄：{ ' / '.join(TICKERS) }\n"
            f"#AI株 #AIN10 #米国株\n"
        )

    print("✅ intraday outputs:")
    print(os.path.abspath(HIS_CSV))
    print(os.path.abspath(IMG_PATH))
    print(os.path.abspath(POST_PATH))


if __name__ == "__main__":
    main()
