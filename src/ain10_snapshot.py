# -*- coding: utf-8 -*-
"""
AIN-10 (AI US) Intraday Snapshot
- 米国AIセクター10社の1日（当日）指数チャート（5分足優先）
- 出力: docs/outputs/ain10_intraday.csv / ain10_intraday.png / ain10_intraday_post.txt
"""

import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ====== 設定 ======
TICKERS = [
    "ORCL", "PLTR", "GOOGL", "MSFT", "CRWV", "META",
    "NVDA", "AMD", "NBIS", "AMZN"
]

OUT_DIR = "docs/outputs"
CSV_PATH = os.path.join(OUT_DIR, "ain10_intraday.csv")
IMG_PATH = os.path.join(OUT_DIR, "ain10_intraday.png")
TXT_PATH = os.path.join(OUT_DIR, "ain10_intraday_post.txt")

PREFERRED_INTERVALS = ["5m", "15m", "30m", "60m"]  # 順にフォールバック
PERIOD = "1d"  # 当日
# ===================


def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def force_series_close(df: pd.DataFrame) -> pd.Series:
    """
    df["Close"] を必ず 1次元 Series(float) にして返す。
    想定外型でも Series に包んで to_numeric で強制変換。
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # yfinance は MultiIndex になることがあるが、単一ティッカーなら基本通常列
    close = df.get("Close", None)
    if close is None:
        return pd.Series(dtype=float)

    # numpy配列/スカラー等でも Series に包む
    if not isinstance(close, (pd.Series, pd.Index)):
        close = pd.Series(close, index=df.index)

    s = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    # index の tz を外す（混在対策）
    try:
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    except Exception:
        pass
    return s


def fetch_intraday_one(ticker: str) -> pd.Series:
    """
    1銘柄を複数 interval で試行し、最初に取れた Series を返す。
    """
    tk = yf.Ticker(ticker)
    for iv in PREFERRED_INTERVALS:
        try:
            df = tk.history(
                period=PERIOD,
                interval=iv,
                prepost=False,
                auto_adjust=False,
                actions=False,
            )
            s = force_series_close(df)
            if not s.empty:
                return s
        except Exception as e:
            print(f"[WARN] fetch failed for {ticker} ({iv}): {e}")
            continue
    return pd.Series(dtype=float)


def build_index(price_df: pd.DataFrame) -> pd.Series:
    """
    等金額（等加重）で当日の変化率を平均し、基準0%からの累積を可視化するため
    当日始値比 (%) を直接平均した Series を返す。
    """
    base = price_df.iloc[0]
    change_pct = (price_df / base - 1.0) * 100.0
    index_series = change_pct.mean(axis=1, skipna=True)
    index_series.name = "AIN-10"
    return index_series


def plot_chart(index_series: pd.Series):
    """当日インデックスのライン色：上昇＝青緑、下落＝赤"""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

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
        s = fetch_intraday_one(t)
        if s.empty:
            print(f"[WARN] No intraday data for {t}")
        else:
            series_map[t] = s

    if not series_map:
        raise RuntimeError("No intraday data for any ticker.")

    # 列方向に結合し、前方補完・全欠損行は落とす
    df = pd.concat(series_map, axis=1).ffill().dropna(how="all")
    # 列MultiIndexの最上位を外して通常列名化（ORCL, PLTR, ...）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # 指数（当日始値比の平均）
    index_series = build_index(df)

    # 保存
    pd.DataFrame(index_series).to_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"[OK] CSV saved: {CSV_PATH}")

    # プロット
    plot_chart(index_series)

    # 投稿用テキスト
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
