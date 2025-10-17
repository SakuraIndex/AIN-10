#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 列名推定（1列目=時刻/日付、2列目=値）
    if df.shape[1] < 2:
        raise ValueError(f"CSV has too few columns: {path}")
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[ts_col, val_col]].rename(columns={ts_col: "ts", val_col: "value"})
    # ts は文字列のままでOK（文字列日付も許容）
    # 数値のみ抽出
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df

def try_prev_close(history_path: Path) -> float | None:
    if not history_path.exists():
        return None
    h = pd.read_csv(history_path)
    if h.shape[1] < 2:
        return None
    # 末尾の値を「前日終値」とみなす（生成ロジックにより日次末尾が並ぶ前提）
    val_col = h.columns[1]
    s = pd.to_numeric(h[val_col], errors="coerce").dropna()
    return float(s.iloc[-2]) if len(s) >= 2 else (float(s.iloc[-1]) if len(s) >= 1 else None)

def safe_pct(num: float, den: float) -> float | None:
    if den is None: return None
    if den == 0 or not np.isfinite(den): return None
    return (num - den) / den * 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True, help="docs/outputs/ain10_1d.csv")
    ap.add_argument("--history", default=None, help="docs/outputs/ain10_history.csv（任意）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    df = read_series(Path(args.csv))
    if df.empty:
        basis = "n/a"; pct = None; valid = "n/a"
    else:
        # その日の最初と最後
        open_val = float(df["value"].iloc[0])
        last_val = float(df["value"].iloc[-1])
        valid = f"{str(df['ts'].iloc[0]).split(' ')[0]} first->latest"

        # しきい値（0 近傍で暴走しないように）
        MIN_DEN = 0.1

        pct = None
        basis = None

        # (1) open が十分大きい
        if abs(open_val) >= MIN_DEN:
            pct = safe_pct(last_val, open_val)
            basis = "open"

        # (2) 前日終値が使えそうなら
        if pct is None and args.history:
            prev_close = try_prev_close(Path(args.history))
            if prev_close is not None and abs(prev_close) >= MIN_DEN:
                pct = safe_pct(last_val, prev_close)
                basis = "prev_close"

        # (3) その日の絶対最大値でスケール（目安比率）
        if pct is None:
            absmax = float(np.nanmax(np.abs(df["value"].to_numpy()))) if len(df) else np.nan
            if np.isfinite(absmax) and absmax >= MIN_DEN:
                pct = (last_val - open_val) / absmax * 100.0
                basis = "absmax"

        # (最終) それでも無理なら N/A
        if pct is None or not np.isfinite(pct):
            pct = None
            basis = "n/a"
            valid = "n/a"

    # 出力テキスト（レベルΔは常に N/A 固定）
    if pct is None:
        line = f"{args.index_key.upper()} 1d: Δ=N/A (level)  A%=N/A (basis={basis} valid={valid})"
        pct_out = None
    else:
        sign_pct = f"{pct:+.2f}%"
        line = f"{args.index_key.upper()} 1d: Δ=N/A (level)  A%={sign_pct} (basis={basis} valid={valid})"
        pct_out = float(pct)

    Path(args.out_text).write_text(line + "\n", encoding="utf-8")

    # stats.json（サイト用；レベル差は常に N/A）
    stats = {
        "index_key": args.index_key,
        "pct_1d": pct_out,
        "delta_level": None,
        "scale": "level",
        "basis": basis,
        "updated_at": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    Path(args.out_json).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
