#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple

# ---------- 時刻ユーティリティ ----------
def iso_now() -> str:
    """常にUTCのISO8601（Z付き）を返す。"""
    ts = pd.Timestamp.utcnow()
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return ts.isoformat().replace("+00:00", "Z")

# ---------- 入力読み込み ----------
def read_1d(csv_path: Path) -> pd.DataFrame:
    """
    docs/outputs/ain10_1d.csv を読み込み、先頭2列を ts/val に正規化。
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df

# ---------- 基準値の決定 ----------
def pick_baseline(
    df: pd.DataFrame,
    basis: str,
    min_abs_for_pct: float = 0.3,
) -> Tuple[Optional[float], str]:
    """
    騰落率の分母となる基準値と、basis 注記文字列を返す。
    - "open"        : 一番最初の値
    - "prev_close"  : 今は open と同じ扱い（将来差し替え可）
    - "stable@HH:MM": 指定時刻以降で最初の非ゼロ（|val|>1e-9）の値
    戻り値: (baseline, basis_note)
    """
    note = basis
    first_val = float(df.iloc[0]["val"])

    if basis.startswith("stable@"):
        try:
            hhmm = basis.split("@", 1)[1]
            hh, mm = map(int, hhmm.split(":"))
        except Exception:
            # フォーマット不正なら open にフォールバック
            return first_val, "open(fallback)"

        # 指定時刻以降の最初の非ゼロ（または最初に |val| が 1e-9 を超える点）
        t0 = df["ts"].dt.normalize().iloc[0] + pd.Timedelta(hours=hh, minutes=mm)
        sub = df[df["ts"] >= t0]
        base_row = sub[abs(sub["val"]) > 1e-9].head(1)
        if base_row.empty:
            # 見つからなければ、全期間の最初の非ゼロを使う
            base_row = df[abs(df["val"]) > 1e-9].head(1)
            note = f"stable@{hhmm}(fallback-first-nonzero)"
        if base_row.empty:
            # それでもなければ open
            return first_val, f"stable@{hhmm}(fallback-open)"
        base_val = float(base_row["val"].iloc[0])
        note = f"stable@{hhmm}"
    elif basis == "open":
        base_val = first_val
        note = "open"
    elif basis == "prev_close":
        # 今回はデータが無いので open と同じ
        base_val = first_val
        note = "prev_close(open)"
    else:
        # 未知指定は open
        base_val = first_val
        note = "open(unknown-basis)"

    # 分母が小さすぎると % が暴れるので抑止
    if abs(base_val) < min_abs_for_pct:
        return None, "no_pct_col"  # 理由を note に入れておく

    return base_val, note

# ---------- % 計算 ----------
def percent_change(base: float, last: float) -> Optional[float]:
    """((last - base) / |base|) * 100"""
    try:
        return (float(last) - float(base)) / abs(float(base)) * 100.0
    except Exception:
        return None

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # open / prev_close / stable@HH:MM
    ap.add_argument("--basis", default="stable@10:00")
    args = ap.parse_args()

    df = read_1d(Path(args.csv))
    pct_val: Optional[float] = None
    delta_level: Optional[float] = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        last_val = float(last_row["val"])
        delta_level = last_val - float(first_row["val"])
        valid_note = f"{first_row['ts']}->{last_row['ts']}"

        base_val, basis_note = pick_baseline(df, args.basis)
        if base_val is not None:
            pct_val = percent_change(base_val, last_val)

    # --- TXT ---
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON ---
    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "level",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
