#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X投稿用の当日騰落率(%)を計算して docs/outputs/{index}_post_intraday.txt と
docs/outputs/{index}_stats.json を更新する。

ポリシー:
- チャートは level のみ。％は出さない（長期・短期どちらも）。
- ％計算は「価格列(close/price)が CSV に存在するときだけ」行う。
  それ以外（レベルしか無い）は A%=N/A のまま。
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = "docs/outputs"

# 取りにいく候補CSV（どちらかが存在すればそれを使う）
CANDIDATE_CSVS = [
    f"{OUT_DIR}/{INDEX_KEY}_intraday.csv",
    f"{OUT_DIR}/{INDEX_KEY}_1d.csv",
]

# 時間列の候補
TS_CANDIDATES = ["ts", "time", "timestamp", "date", "datetime", "Datetime"]

# 価格列の候補（これが無ければ％は計算しない）
PRICE_CANDIDATES = ["close", "price", "Close", "Price"]

# 参考: レベル差は長期チャートで使う（ここでもヘッダに載せるだけ）
LEVEL_DELTA_SOURCE = f"{OUT_DIR}/{INDEX_KEY}_1d.csv"
LEVEL_COL_CANDIDATES = ["level", "value", "index", INDEX_KEY.upper(), INDEX_KEY]


def _find_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    lower_cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None


def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # indexは使わない。列名の前後空白を除去
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_series_for_price() -> tuple[pd.Series | None, pd.Series | None, str | None]:
    """価格の％計算に使う Series を返す (t, price, csv_path)"""
    for p in CANDIDATE_CSVS:
        df = _read_csv(p)
        if df is None:
            continue
        tcol = _find_col(df, TS_CANDIDATES)
        pcol = _find_col(df, PRICE_CANDIDATES)
        if tcol and pcol:
            ts = pd.to_datetime(df[tcol])
            price = pd.to_numeric(df[pcol], errors="coerce")
            mask = price.notna()
            if mask.sum() >= 2:
                return ts[mask].reset_index(drop=True), price[mask].reset_index(drop=True), p
    return None, None, None


def _calc_level_delta() -> float | None:
    """レベル差(最終 - 先頭)を 1d CSV 等から求める（ヘッダの参考値）。"""
    df = _read_csv(LEVEL_DELTA_SOURCE)
    if df is None:
        return None
    lcol = _find_col(df, LEVEL_COL_CANDIDATES)
    tcol = _find_col(df, TS_CANDIDATES)
    if not lcol or not tcol:
        return None
    s = pd.to_numeric(df[lcol], errors="coerce")
    ts = pd.to_datetime(df[tcol], errors="coerce")
    mask = s.notna() & ts.notna()
    s = s[mask]
    if len(s) < 2:
        return None
    return float(s.iloc[-1] - s.iloc[0])


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) まずレベル差（参考）
    delta_level = _calc_level_delta()

    # 2) 価格が取れるなら当日％を計算
    ts, price, used_csv = _load_series_for_price()
    pct_1d: float | None = None
    basis = "n/a"

    if ts is not None and price is not None:
        first = float(price.iloc[0])
        last = float(price.iloc[-1])
        if first != 0:
            pct_1d = (last / first - 1.0) * 100.0
            basis = "open"  # 「当日最初の価格に対する騰落率」
        else:
            pct_1d = None
            basis = "n/a"

    # 3) テキスト出力（X用の下書き）
    now_utc = datetime.now(timezone.utc)
    valid_range = "n/a"
    if ts is not None:
        valid_range = f"{ts.iloc[0].strftime('%Y-%m-%d %H:%M:%S')}->{ts.iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}"

    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    pct_str = "N/A" if pct_1d is None else f"{pct_1d:+.2f}%"

    txt_line = (
        f"{INDEX_KEY.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis {basis} valid={valid_range})"
    )

    with open(f"{OUT_DIR}/{INDEX_KEY}_post_intraday.txt", "w", encoding="utf-8") as f:
        f.write(txt_line + "\n")

    # 4) JSON（サイトが読む指標）も更新
    stats_path = f"{OUT_DIR}/{INDEX_KEY}_stats.json"
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None else float(round(pct_1d, 6)),
        "delta_level": None if delta_level is None else float(round(delta_level, 6)),
        "scale": "level",          # チャートは level 固定
        "basis": basis,            # ％の基準（open or n/a）
        "updated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # 5) デバッグ出力
    print(f"[make_intraday_post] wrote: {stats_path}")
    print(f"[make_intraday_post] wrote: {OUT_DIR}/{INDEX_KEY}_post_intraday.txt")
    if used_csv:
        print(f"[make_intraday_post] price source: {used_csv}")
    else:
        print("[make_intraday_post] price source: <none> (A%=N/A)")


if __name__ == "__main__":
    main()
