#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
import pandas as pd


def iso_now() -> str:
    """UTCのISO8601文字列（Z付き）"""
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")


def read_1d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    return df


def pick_baseline(df: pd.DataFrame, basis: str) -> tuple[float | None, str]:
    """
    basis:
      - "open"       : 最初の行
      - "prev_close" : 最初の行（将来拡張。今は open と同じ扱い）
      - "stable@HH:MM": その日の HH:MM 直近（前後±10分）から最初に見つかった値
      - その他       : 先頭（open）
    """
    basis_note = basis

    if df.empty:
        return None, "n/a"

    if basis.startswith("stable@"):
        try:
            hhmm = basis.split("@", 1)[1]
            target = pd.to_datetime(df["ts"].dt.strftime("%Y-%m-%d") + " " + hhmm)
            # その日の同時刻ターゲット（行ごとに日の一致を利用）
            # 近傍±10分で一番近いものを採用
            day = df["ts"].dt.floor("D").max()
            day_df = df[df["ts"].dt.floor("D") == day]
            if day_df.empty:
                # 予備：全体から
                cand = df.copy()
            else:
                cand = day_df.copy()

            t0 = pd.Timestamp(day.strftime("%Y-%m-%d") + f" {hhmm}")
            cand["abs_diff"] = (cand["ts"] - t0).abs()
            cand = cand[cand["abs_diff"] <= pd.Timedelta(minutes=10)]
            if not cand.empty:
                v = float(cand.sort_values("abs_diff").iloc[0]["val"])
                return v, basis
            # 見つからない場合は open にフォールバック
            return float(df.iloc[0]["val"]), f"{basis}(fallback=open)"
        except Exception:
            return float(df.iloc[0]["val"]), f"{basis}(fallback=open)"

    if basis in ("open", "prev_close", "first"):
        return float(df.iloc[0]["val"]), basis

    # デフォルト：open
    return float(df.iloc[0]["val"]), "open"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/*_1d.csv")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    # 比率ではなく差分%を採用
    ap.add_argument("--basis", default="open")  # open / prev_close / stable@HH:MM
    args = ap.parse_args()

    df = read_1d(Path(args.csv))

    pct_val: float | None = None
    delta_level: float | None = None
    basis_note = "n/a"
    valid_note = "n/a"

    if not df.empty:
        base, basis_note = pick_baseline(df, args.basis)
        last_row = df.iloc[-1]
        if base is not None and not (pd.isna(base) or pd.isna(last_row["val"])):
            last_val = float(last_row["val"])
            # 差分で評価（×100 で % に）
            delta_level = last_val - float(base)
            pct_val = delta_level * 100.0
            valid_note = f"{df.iloc[0]['ts']}->{last_row['ts']}"

    # --- TXT 出力
    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    text = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} valid={valid_note})\n"
    )
    Path(args.out_text).write_text(text, encoding="utf-8")

    # --- JSON 出力
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
