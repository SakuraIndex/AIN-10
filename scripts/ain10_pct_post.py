#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1日騰落率(%)とレベル差を算出し、
- docs/outputs/<index_key>_post_intraday.txt
- docs/outputs/<index_key>_stats.json
を更新する小ツール。

特徴:
- CSVのカラム名が固定でなくてもOK（先頭列=日時、2列目=値として解釈）
- オープン値が0や極小で%が出せないときは、直近の非ゼロ値へ自動フォールバック
- データが欠損・全ゼロの場合は A%=N/A として安全に終了
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_intraday_csv(csv_path: Path) -> pd.DataFrame:
    """先頭列=時刻, 2列目=値 とみなして読み込む。"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV needs at least 2 columns: {csv_path}")
    # 先頭列を時刻に、2列目を値に正規化
    ts_col = df.columns[0]
    val_col = df.columns[1]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], errors="coerce"),
        "val": pd.to_numeric(df[val_col], errors="coerce")
    }).dropna(subset=["ts"])
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def first_non_nan(series: pd.Series) -> float | None:
    """最初の非NaN値を返す。なければNone。"""
    s = series.dropna()
    return None if s.empty else float(s.iloc[0])


def last_non_nan(series: pd.Series) -> float | None:
    """最後の非NaN値を返す。なければNone。"""
    s = series.dropna()
    return None if s.empty else float(s.iloc[-1])


def find_first_nonzero(series: pd.Series, scan: int = 20, eps: float = 1e-9) -> float | None:
    """
    先頭からscan件のうち、ゼロでない(絶対値>eps)最初の値を探す。
    見つからなければNone。
    """
    s = series.dropna()
    if s.empty:
        return None
    head = s.iloc[:scan]
    nz = head[head.abs() > eps]
    return None if nz.empty else float(nz.iloc[0])


def compute_change_and_pct(df: pd.DataFrame) -> tuple[float | None, float | None, str, str]:
    """
    Δlevel と pct(%) を計算。
    戻り値: (delta_level, pct_1d, basis, valid_note)
      - basis: "open" か "first_nonzero" か "n/a"
      - valid_note: "first->latest" 等、簡易バリデーション文字列
    """
    if df.empty:
        return None, None, "n/a", "n/a"

    open_val = first_non_nan(df["val"])
    last_val = last_non_nan(df["val"])
    if open_val is None or last_val is None:
        return None, None, "n/a", "n/a"

    delta = last_val - open_val

    # デフォルト: open基準
    basis = "open"
    base_val = open_val

    # open==0（または極小）ならフォールバック
    if base_val is None or abs(base_val) < 1e-9:
        alt = find_first_nonzero(df["val"], scan=30, eps=1e-9)
        if alt is not None:
            basis = "first_nonzero"
            base_val = alt
        else:
            # %は算出不能
            return delta, None, "n/a", "first->latest"

    pct = (last_val - base_val) / abs(base_val) * 100.0
    return delta, pct, basis, "first->latest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True, help="例: ain10 / astra4 / ...")
    ap.add_argument("--csv", required=True, help="docs/outputs/<index>_1d.csv")
    ap.add_argument("--out-json", required=True, help="docs/outputs/<index>_stats.json")
    ap.add_argument("--out-text", required=True, help="docs/outputs/<index>_post_intraday.txt")
    args = ap.parse_args()

    index_key = args.index_key.strip()
    csv_path = Path(args.csv)
    out_json = Path(args.out_json)
    out_text = Path(args.out_text)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = load_intraday_csv(csv_path)
    except Exception as e:
        # 入力が壊れている等 → セーフにN/Aで吐いて終了
        msg = f"{index_key.upper()} 1d: Δ=N/A (level) A%=N/A (basis=n/a valid=n/a)"
        out_text.write_text(msg + "\n", encoding="utf-8")
        payload = {
            "index_key": index_key,
            "pct_1d": None,
            "delta_level": None,
            "scale": "level",
            "basis": "n/a",
            "updated_at": _utcnow_iso(),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    delta, pct, basis, valid_note = compute_change_and_pct(df)

    # テキスト出力
    if delta is None:
        delta_str = "N/A"
    else:
        delta_str = f"{delta:.6f}".rstrip("0").rstrip(".")

    if pct is None:
        pct_str = "N/A"
    else:
        pct_str = f"{pct:+.2f}%"

    line = f"{index_key.upper()} 1d: Δ={delta_str} (level) A%={pct_str} (basis={basis} valid={valid_note})"
    out_text.write_text(line + "\n", encoding="utf-8")

    # JSON出力
    payload = {
        "index_key": index_key,
        "pct_1d": None if pct is None else float(pct),
        "delta_level": None if delta is None else float(delta),
        "scale": "level",
        "basis": basis if pct is not None else "n/a",
        "updated_at": _utcnow_iso(),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
