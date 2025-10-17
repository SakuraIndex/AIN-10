# scripts/ain10_pct_post.py
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

TIME_CANDIDATES = ["ts","time","timestamp","date","datetime","Datetime"]

def load_first_last(csv_path: str) -> tuple[float | None, float | None, str]:
    df = pd.read_csv(csv_path)
    # 推定: 時刻列
    time_col = None
    for c in TIME_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    # 値列（最後の非時刻列を採用）
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        return None, None, "n/a"
    vcol = value_cols[-1]

    # 時系列整形
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.dropna(subset=[vcol])
    if len(df) == 0:
        return None, None, "n/a"

    first = pd.to_numeric(df[vcol], errors="coerce").dropna()
    if first.empty:
        return None, None, "n/a"
    first_val = float(first.iloc[0])
    last_val  = float(first.iloc[-1])
    valid = df[time_col].dt.tz_convert(None) if df[time_col].dt.tz is not None else df[time_col]
    valid_date = valid.iloc[0].date().isoformat()
    return first_val, last_val, valid_date

def pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None:
        return None
    if first == 0:
        return None
    return (last - first) / abs(first) * 100.0

def fmt_pct(p: float | None) -> str:
    if p is None:
        return "N/A"
    s = f"{p:+.2f}%"
    # ±0.00% は +0.00% と同等扱いに正規化
    if s in ["-0.00%", "+0.00%"]:
        s = "+0.00%"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--basis", default="open", choices=["open","prev_close"])
    args = ap.parse_args()

    first, last, valid_date = load_first_last(args.csv)
    p = pct_change(first, last)

    # X投稿用テキスト（レベル差分Δは本件では常時N/A固定）
    line = (
        f"{args.index_key.upper()} 1d: Δ=N/A (level) "
        f"A%={fmt_pct(p)} (basis={args.basis} valid={valid_date} first->latest)"
    )
    Path(args.out_text).write_text(line + "\n", encoding="utf-8")

    # stats.json を軽量更新
    now = datetime.now(timezone.utc).isoformat()
    obj = {
        "index_key": args.index_key,
        "pct_1d": None if p is None else float(p),
        "delta_level": None,
        "scale": "level",
        "basis": args.basis,
        "updated_at": now,
    }
    Path(args.out_json).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
