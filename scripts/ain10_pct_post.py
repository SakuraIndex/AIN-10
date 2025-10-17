# scripts/ain10_pct_post.py
# -*- coding: utf-8 -*-
import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

INDEX_KEY = os.environ.get("INDEX_KEY", "ain10")
OUT_DIR = Path("docs/outputs")
CSV_1D = OUT_DIR / f"{INDEX_KEY}_1d.csv"
POST_TXT = OUT_DIR / f"{INDEX_KEY}_post_intraday.txt"
STATS_JSON = OUT_DIR / f"{INDEX_KEY}_stats.json"

# 列名の候補
DT_CANDIDATES = ["ts", "time", "timestamp", "date", "datetime", "Datetime", "Date", "Time"]
VAL_CANDIDATES = [
    "value", "index", "score", "close", "price", "level",
    INDEX_KEY, INDEX_KEY.upper(), INDEX_KEY.replace("-", ""),
    INDEX_KEY.replace("-", "_"), INDEX_KEY.replace("_", "-").upper(),
    "AIN-10", "AIN10", "AIN_10"
]

def _pick_datetime_col(df: pd.DataFrame) -> str:
    # 候補一致 or datetime型推定
    for c in DT_CANDIDATES:
        if c in df.columns:
            return c
    # 型から推測
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    raise ValueError(f"datetime列が見つかりません: columns={list(df.columns)}")

def _pick_value_col(df: pd.DataFrame, dt_col: str) -> str:
    # 明示候補
    for c in VAL_CANDIDATES:
        if c in df.columns:
            return c
    # 数値列の中から dt_col 以外で最右を採用
    numeric_cols = [c for c in df.columns if c != dt_col and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]
    # 数値に変換できる列を探索
    for c in df.columns:
        if c == dt_col:
            continue
        try:
            pd.to_numeric(df[c])
            return c
        except Exception:
            continue
    raise ValueError(f"value列が見つかりません: columns={list(df.columns)}")

def load_series(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    dt_col = _pick_datetime_col(df)
    val_col = _pick_value_col(df, dt_col)

    # 前処理
    df = df[[dt_col, val_col]].copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna().sort_values(dt_col)
    if df.empty:
        raise ValueError("有効データが空です")

    return df.rename(columns={dt_col: "ts", val_col: "value"})

def compute_open_to_last_pct(df: pd.DataFrame) -> tuple[float, str]:
    # その日の先頭(始値)と末尾(最新)
    open_row = df.iloc[0]
    last_row = df.iloc[-1]
    open_v = float(open_row["value"])
    last_v = float(last_row["value"])
    if open_v == 0:
        pct = 0.0
    else:
        pct = (last_v / open_v - 1.0) * 100.0
    # バリデーションの表記（始値のタイムスタンプ日付）
    basis_date = open_row["ts"].strftime("%Y-%m-%d")
    return pct, basis_date

def read_stats(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 既定骨組み
    return {
        "index_key": INDEX_KEY,
        "pct_1d": None,
        "delta_level": None,
        "scale": "level",
        "basis": "n/a",
        "updated_at": None,
    }

def main():
    df = load_series(CSV_1D)
    pct, basis_date = compute_open_to_last_pct(df)

    # ポスト文面（レベルのΔは出さない = 仕様）
    line = (
        f"{INDEX_KEY.upper()} 1d: "
        f"A%={'+' if pct>=0 else ''}{pct:.2f}% "
        f"(basis=open valid={basis_date} open -> latest)"
    )
    POST_TXT.write_text(line + "\n", encoding="utf-8")
    print(f"wrote {POST_TXT} : {line}")

    # stats.json 更新（既存を尊重し、pct_1d / basis / updated_at だけ上書き）
    stats = read_stats(STATS_JSON)
    stats["index_key"] = INDEX_KEY
    stats["pct_1d"] = round(pct, 6)
    stats["basis"] = "open"
    stats["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    print(f"updated {STATS_JSON} : pct_1d={stats['pct_1d']} basis=open")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ain10_pct_post] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
