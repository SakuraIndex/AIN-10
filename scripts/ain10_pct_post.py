# scripts/ain10_pct_post.py
import argparse
import json
from pathlib import Path
import math
import pandas as pd

TIME_CANDIDATES = ["Datetime", "datetime", "timestamp", "time", "date", "Date", "Timestamp"]

def pick_time_column(df: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c
    # インデックスが日時ならそれを使う
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        return "index"
    raise ValueError(f"Time column not found. Columns={list(df.columns)}")

def pick_value_column(df: pd.DataFrame, index_key: str, time_col: str) -> str:
    # INDEX_KEY と一致/関連しそうな列を優先（大文字・ハイフン差異も吸収）
    key_variants = {
        index_key.lower(),
        index_key.upper(),
        index_key.replace("_", "-").upper(),
        index_key.replace("-", "_").lower(),
    }
    for c in df.columns:
        if c == time_col:
            continue
        cn = str(c)
        if cn.lower() in key_variants or cn.upper() in key_variants:
            return c
        # AIN-10 のようにハイフンを含むケース
        if cn.replace("_", "-").lower() in key_variants:
            return c
        if cn.replace("-", "_").lower() in key_variants:
            return c

    # 数値列のうち最後の列を採用（時間列は除外）
    numeric_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric value column found.")
    return numeric_cols[-1]

def safe_pct(latest: float, base: float):
    if base is None or latest is None:
        return None
    if isinstance(base, float) and (math.isnan(base) or base == 0.0):
        return None
    if isinstance(latest, float) and math.isnan(latest):
        return None
    pct = (latest / base - 1.0) * 100.0
    # 明らかな外れ値を弾く（データ欠損の可能性）
    if abs(pct) > 50.0:
        return None
    return pct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    args = ap.parse_args()

    index_key = args.index_key
    csv_path = Path(args.csv)
    out_json = Path(args.out_json)
    out_text = Path(args.out_text)

    if not csv_path.exists():
        # 生成物がないときは N/A で書く
        msg = f"{index_key.upper()} 1d: A%=N/A (basis=n/a valid=n/a)"
        out_text.write_text(msg + "\n", encoding="utf-8")
        out_json.write_text(json.dumps({
            "index_key": index_key,
            "pct_1d": None,
            "delta_level": None,
            "scale": "level",
            "basis": "n/a",
            "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
        }), encoding="utf-8")
        return

    df = pd.read_csv(csv_path)
    # 時間列の抽出
    time_col = pick_time_column(df)
    if time_col == "index":
        df["time"] = pd.to_datetime(df["index"])
        time_col = "time"
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # 値列の抽出
    val_col = pick_value_column(df, index_key, time_col)

    # 時間でソート & 最新日を特定
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        basis = "n/a"
        pct = None
    else:
        latest_ts = df[time_col].iloc[-1]
        latest_date = latest_ts.date()

        # 最新日の「最初の行」を open とする（取引所の正確な 09:30 が無くても、その日の最初でOK）
        day_mask = df[time_col].dt.date == latest_date
        day_df = df.loc[day_mask]
        if len(day_df) >= 2:
            open_val = day_df[val_col].iloc[0]
            latest_val = day_df[val_col].iloc[-1]
            pct = safe_pct(latest_val, open_val)
            basis = "open"
            valid = f"{latest_date} first->latest"
        else:
            # その日のデータが 1本しかない/ない → N/A（無理に prev_any は使わない）
            pct = None
            basis = "n/a"
            valid = "n/a"

    # 出力
    if pct is None:
        pct_str = "N/A"
    else:
        sign = "+" if pct >= 0 else ""
        pct_str = f"{sign}{pct:.2f}%"

    # テキスト
    text = f"{index_key.upper()} 1d: A%={pct_str} (basis={basis} valid={valid})"
    out_text.write_text(text + "\n", encoding="utf-8")

    # JSON（サイト側の取り回し用）
    out_json.write_text(json.dumps({
        "index_key": index_key,
        "pct_1d": None if pct is None else float(pct),
        "delta_level": None,     # レベルは表示しない方針
        "scale": "level",
        "basis": basis,
        "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }), encoding="utf-8")

if __name__ == "__main__":
    main()
