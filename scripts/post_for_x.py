#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X（旧Twitter）投稿用の「1日騰落率（%）」だけを計算して
docs/outputs/<key>_intraday_post.txt（なければ <key>_post.txt）へ書き出す。
レベル系列の内部出力（PNG/CSV/JSON）は一切変更しない（ASTRA4準拠）。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import csv
from datetime import datetime, timezone, date

# ルート/出力ディレクトリ（現在のリポ構成に合わせてあります）
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "docs" / "outputs"


# ---------- ユーティリティ ----------

def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _parse_date(s: str) -> Optional[date]:
    """
    'YYYY-MM-DD' or ISO8601日時から日付だけを取り出す。
    """
    s = str(s).strip()
    if not s:
        return None
    # 素直に日付だけの場合
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except Exception:
            pass
    # ISO 8601 datetime のとき
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _find_first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


# ---------- データ読み込み ----------

@dataclass
class IntradayData:
    times: List[str]
    values: List[float]

def load_intraday_csv(key: str) -> IntradayData:
    """
    想定ファイル: docs/outputs/<key>_1d.csv
    カラム名が環境でバラついても読めるように、値カラムは
    ['value','close','price','index','level'] の優先順で探索。
    時刻カラムは ['time','timestamp','datetime','date'] の優先順で探索。
    """
    path = OUT / f"{key}_1d.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing intraday csv: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"empty intraday csv: {path}")

    # 値カラム推定
    value_candidates = ["value", "close", "price", "index", "level"]
    time_candidates  = ["time", "timestamp", "datetime", "date"]

    header = rows[0].keys()
    vcol = next((c for c in value_candidates if c in header), None)
    tcol = next((c for c in time_candidates  if c in header), None)

    if vcol is None:
        # 最右列を値とみなすフォールバック
        vcol = list(header)[-1]
    if tcol is None:
        # 最左列を時刻とみなすフォールバック
        tcol = list(header)[0]

    times  = [str(r[tcol]) for r in rows]
    values = []
    for r in rows:
        v = _to_float(r.get(vcol))
        if v is None:
            raise ValueError(f"non-numeric value in intraday csv: {r.get(vcol)}")
        values.append(v)

    return IntradayData(times=times, values=values)


@dataclass
class DailyHistory:
    dates: List[date]
    closes: List[float]

def load_history_csv(key: str) -> DailyHistory:
    """
    想定ファイル: docs/outputs/<key>_history.csv
    カラム名は ['date','close'] を推奨。
    もし無ければ、最左列を日付、最右列を終値として解釈する。
    無ければ空で返す。
    """
    path = OUT / f"{key}_history.csv"
    if not path.exists():
        return DailyHistory(dates=[], closes=[])

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return DailyHistory(dates=[], closes=[])

    header = rows[0].keys()
    dcol = "date"  if "date"  in header else list(header)[0]
    ccol = "close" if "close" in header else list(header)[-1]

    dates: List[date] = []
    closes: List[float] = []
    for r in rows:
        d = _parse_date(str(r.get(dcol)))
        c = _to_float(r.get(ccol))
        if d is not None and c is not None:
            dates.append(d)
            closes.append(c)

    return DailyHistory(dates=dates, closes=closes)


# ---------- 計算ロジック ----------

@dataclass
class PctResult:
    pct_str: str    # "+0.42%" または "N/A"
    basis:   str    # "prev_close" | "open" | "n/a"

def compute_1d_pct_for_post(intra: IntradayData, hist: DailyHistory) -> PctResult:
    """
    原則: 前日終値（prev_close）基準。
    取得できない場合は「当日寄り（open）」基準にフォールバック。
    異常値（±15%超）や計算不可能な場合は "N/A"。
    """
    last = intra.values[-1]
    open_ = intra.values[0]

    # intraday の日付（最初の行のタイムスタンプから推定）
    today = _parse_date(intra.times[0])

    prev_close: Optional[float] = None
    if hist.dates and hist.closes and today is not None:
        # today より前で最大の日付の終値
        prevs = [(d, c) for d, c in zip(hist.dates, hist.closes) if d < today]
        if prevs:
            prev_close = prevs[-1][1]  # 末尾は直近日

    def _format(p: Optional[float]) -> PctResult:
        if p is None:
            return PctResult("N/A", "n/a")
        if abs(p) > 15.0:  # 安全弁
            return PctResult("N/A", "prev_close")
        return PctResult(f"{p:+.2f}%", "prev_close")

    if prev_close and prev_close != 0:
        pct = (last / prev_close - 1.0) * 100.0
        r = _format(pct)
        # basis の明記
        return PctResult(r.pct_str if r.pct_str != "N/A" else "N/A", "prev_close")

    # フォールバック: 当日寄り
    if open_ and open_ != 0:
        pct = (last / open_ - 1.0) * 100.0
        if abs(pct) <= 15.0:
            return PctResult(f"{pct:+.2f}%", "open")
        else:
            return PctResult("N/A", "open")

    return PctResult("N/A", "n/a")


# ---------- 出力 ----------

def write_post_text(key: str, res: PctResult) -> Path:
    """
    既存のポスト用ファイルがあればそれを使い、無ければ _intraday_post.txt を新規作成。
    例: docs/outputs/ain10_post_intraday.txt または ain10_post.txt
    """
    candidates = [
        OUT / f"{key}_post_intraday.txt",
        OUT / f"{key}_intraday_post.txt",
        OUT / f"{key}_post.txt",
    ]
    path = _find_first_existing(candidates)
    line = f"{key.upper()} 1d: A%={res.pct_str} (basis {res.basis})\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line, encoding="utf-8")
    return path


# ---------- エントリポイント ----------

def main():
    ap = argparse.ArgumentParser(description="Make 1d percent change for X post (ASTRA4-friendly).")
    ap.add_argument("--key", default="ain10", help="index key, e.g., ain10 / astra4 ...")
    args = ap.parse_args()

    intra = load_intraday_csv(args.key)
    hist  = load_history_csv(args.key)
    res   = compute_1d_pct_for_post(intra, hist)
    outp  = write_post_text(args.key, res)

    # ログ表示（CIの可読性用）
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"[{now}] wrote X-post line to: {outp}")
    print(f"line: {args.key.upper()} 1d: A%={res.pct_str} (basis {res.basis})")


if __name__ == "__main__":
    main()
