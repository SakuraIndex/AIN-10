#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

JST = "Asia/Tokyo"

# -------------------- helpers (datetime) --------------------

def _try_parse_col_as_datetime(s: pd.Series) -> pd.Series:
    """
    CSVの時刻列をJSTのDatetimeIndexへ。
    - まずUTCとしてパース（utc=True）。timezone付ならそのまま扱い、JSTへ変換。
    - 失敗する/naiveの場合はJSTローカライズで最後にJSTへ統一。
    """
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if dt.dt.tz is None:
        # utc=TrueでもNoneになる場合は手動でローカライズ
        dt = pd.to_datetime(s, errors="coerce")
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(JST)
        else:
            dt = dt.dt.tz_convert(JST)
    else:
        dt = dt.dt.tz_convert(JST)
    return dt


def _autodetect_dt_col(raw: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in raw.columns:
            if pd.to_datetime(raw[c], errors="coerce").notna().mean() >= 0.8:
                return c
    best_col, best_valid = None, -1.0
    for c in raw.columns:
        v = pd.to_datetime(raw[c], errors="coerce").notna().mean()
        if v > best_valid:
            best_col, best_valid = c, v
    return best_col if best_valid >= 0.8 else None


def to_jst_index(raw: pd.DataFrame, dt_col_opt: Optional[str]) -> pd.DataFrame:
    candidates = ["Datetime", "datetime", "Timestamp", "timestamp", "Date", "date", "Time", "time"]
    dt_col = dt_col_opt if (dt_col_opt and dt_col_opt in raw.columns) else _autodetect_dt_col(raw, candidates)
    if dt_col is None:
        raise ValueError(f"CSV内の日時列を自動検出できませんでした。--dt-col で明示してください。列={list(raw.columns)}")
    dt = _try_parse_col_as_datetime(raw[dt_col])
    out = raw.copy()
    out.index = dt
    out = out.drop(columns=[dt_col])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _mk_ts(date: pd.Timestamp, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date.date()} {hm}", tz=JST)


def filter_session_cross(df_jst: pd.DataFrame, start_hm: str, end_hm: str, anchor_hm: str) -> pd.DataFrame:
    """
    セッション抽出（跨ぎ対応）。
    - アンカー(anchor_hm)を基準に「営業日」を決める。
    - end_hm が start_hm より早い場合は「翌日」扱いとして切り出し。
    例）US株: start=22:30, end=05:00, anchor=22:30
    """
    if df_jst.empty:
        return df_jst

    last_ts = df_jst.index[-1]
    # 直近データの「アンカー基準日」を決める
    # 最も近い anchor_hm を過去側に取る
    anchor_today = _mk_ts(last_ts, anchor_hm)
    anchor = anchor_today if last_ts >= anchor_today else anchor_today - pd.Timedelta(days=1)

    start = _mk_ts(anchor, start_hm)
    end = _mk_ts(anchor, end_hm)
    if end <= start:
        end = end + pd.Timedelta(days=1)

    return df_jst.loc[(df_jst.index >= start) & (df_jst.index <= end)]


# -------------------- helpers (value column) --------------------

def _variations(key: str) -> List[str]:
    cand = [key, key.lower(), key.upper(), key.capitalize()]
    cand += [f"{x}_mean" for x in cand]
    return list(dict.fromkeys(cand))  # unique, keep order


def find_value_column(df: pd.DataFrame, index_key: str) -> str:
    cols = list(df.columns)
    # 1) 厳密／大小文字違い／_mean 付きの総当たり
    for name in _variations(index_key):
        if name in cols:
            return name
    # 2) "_mean" を含む列がちょうど1つならそれを採用
    mean_like = [c for c in cols if "_mean" in c.lower()]
    if len(mean_like) == 1:
        return mean_like[0]
    # 3) 数値列が1つだけならそれを採用（日時列は既に index 化済み）
    numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    # 4) それもダメなら、数値列の等加重平均を自作するための目印として空文字を返す
    return ""


def _detect_unit_is_ratio(s: pd.Series) -> bool:
    """95%点の絶対値が 0.5 未満なら ratio（= 0.5% 未満がほとんど）とみなす"""
    arr = pd.to_numeric(s, errors="coerce").to_numpy()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return False
    return float(np.quantile(np.abs(arr), 0.95)) < 0.5


def to_percent_series(df: pd.DataFrame, index_key: str, value_type: str) -> Tuple[pd.Series, str]:
    """
    ％系列に揃える。
    - index_key に合致する列があればそれを使用
    - なければ数値列の等加重平均を使用
    - value_type='auto' なら ratio/percent を自動判定
    """
    value_col = find_value_column(df, index_key)
    if value_col == "":
        num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
        if not num_cols:
            raise ValueError("％化できる数値列が見つかりません。")
        s = df[num_cols].mean(axis=1, skipna=True)
    else:
        s = pd.to_numeric(df[value_col], errors="coerce")

    vt = (value_type or "auto").lower()
    if vt == "auto":
        vt = "ratio" if _detect_unit_is_ratio(s) else "percent"
    if vt == "ratio":
        s = s * 100.0

    s = s.dropna()
    s = s.clip(lower=-50.0, upper=50.0)  # 念のため
    return s, (value_col if value_col else "(mean)")


def make_title_label(index_key: str, label: Optional[str]) -> str:
    return label if label else index_key.upper()


# -------------------- CLI --------------------

@dataclass
class Args:
    index_key: str
    csv: str
    out_json: str
    out_text: str
    snapshot_png: str
    session_start: str
    session_end: str
    day_anchor: str
    basis: str
    label: Optional[str]
    dt_col: Optional[str]
    value_type: str  # auto | percent | ratio

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", choices=["auto", "ratio", "percent"], default="auto")
    a = p.parse_args()
    return Args(
        index_key=a.index_key, csv=a.csv, out_json=a.out_json, out_text=a.out_text,
        snapshot_png=a.snapshot_png, session_start=a.session_start, session_end=a.session_end,
        day_anchor=a.day_anchor, basis=a.basis, label=a.label, dt_col=a.dt_col, value_type=a.value_type
    )

# -------------------- Core --------------------

def summarize_and_plot(
    s_pct: pd.Series,
    title_label: str,
    basis_label: str,
    snapshot_png: str,
) -> Tuple[float, pd.Timestamp]:

    if s_pct.empty:
        raise ValueError("セッション内データがありません。")

    last_pct = float(np.round(s_pct.iloc[-1], 4))
    last_ts = s_pct.index[-1]

    # 線色：プラス→青 / マイナス→赤
    line_color = "#00E5FF" if last_pct >= 0 else "#FF4D4D"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")

    ax.plot(s_pct.index, s_pct.values, linewidth=2.0, color=line_color, label=title_label)
    ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#DDDDDD")

    ax.set_title(
        f"{title_label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})  {last_pct:+.2f}%",
        color="#DDDDDD"
    )
    ax.set_xlabel("Time", color="#BBBBBB")
    ax.set_ylabel("Change vs Prev Close (%)", color="#BBBBBB")
    ax.tick_params(colors="#BBBBBB")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(snapshot_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return last_pct, last_ts

def main() -> None:
    args = parse_args()

    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)

    # US市場：22:30-05:00等の跨ぎに対応
    df_sess = filter_session_cross(df, args.session_start, args.session_end, args.day_anchor)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    s_pct, value_col = to_percent_series(df_sess, args.index_key, args.value_type)

    title_label = make_title_label(args.index_key, args.label)
    last_pct, last_ts = summarize_and_plot(
        s_pct, title_label, args.basis, args.snapshot_png
    )

    sign_arrow = "▲" if last_pct >= 0 else "▼"
    lines = [
        f"{sign_arrow} {title_label} 日中スナップショット ({last_ts.tz_convert(JST):%Y/%m/%d %H:%M})",
        f"{last_pct:+.2f}% (基準: {args.basis})",
        f"#{title_label} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON は ratio（小数）で出力（サイト側と揃える）
    stats = {
        "index_key": args.index_key,
        "label": title_label,
        "pct_intraday": float(np.round(last_pct / 100.0, 6)),  # 例: -0.27% → -0.0027
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "value_source": value_col,
        "updated_at": f"{pd.Timestamp.now(tz=JST).isoformat()}",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
