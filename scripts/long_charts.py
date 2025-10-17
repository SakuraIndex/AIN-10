# scripts/long_charts.py
import os
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def guess_time_col(cols):
    cands = ["ts", "time", "timestamp", "date", "datetime", "Datetime"]
    for c in cands:
        for col in cols:
            if col.lower() == c.lower():
                return col
    # fallback: 最初の列を時間とみなす
    return cols[0]

def guess_value_col(df, index_key):
    # index_key にマッチする列を優先
    for col in df.columns:
        if col.lower() == index_key.lower():
            return col
        if index_key.lower() in col.lower():
            return col
    # 数値列のうち、時間列以外の最初を使う
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # もし型推定が効いていない場合、float 変換を試みる
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
                num_cols.append(c)
            except Exception:
                pass
    return num_cols[0] if num_cols else df.columns[-1]

def load_series(csv_path, index_key):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty csv: {csv_path}")
    tcol = guess_time_col(df.columns)
    vcol = guess_value_col(df.drop(columns=[tcol], errors="ignore"), index_key)
    # 時刻パース
    try:
        df[tcol] = pd.to_datetime(df[tcol])
    except Exception:
        # 数値 epoch の可能性
        df[tcol] = pd.to_datetime(df[tcol], unit="s", errors="coerce")
    df = df[[tcol, vcol]].dropna()
    df = df.sort_values(tcol)
    df = df.rename(columns={tcol: "ts", vcol: "value"})
    return df

def to_utc_iso(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------------------------------------------------------
# Rendering (dark theme, red line, no % on chart)
# -----------------------------------------------------------------------------
def render_level_chart(csv_path, png_path, title, index_key):
    df = load_series(csv_path, index_key)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(df["ts"], df["value"], linewidth=2.0, color="#ff6b6b")  # 赤
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    # 当日レベル差（始値→終値）を返す
    delta_level = float(np.round(df["value"].iloc[-1] - df["value"].iloc[0], 6))
    # 有効期間（最初と最後）も返す
    valid = (df["ts"].iloc[0], df["ts"].iloc[-1])
    return delta_level, valid

# -----------------------------------------------------------------------------
# Stats writers
# -----------------------------------------------------------------------------
def write_post_intraday_txt(path, index_key, delta_level, valid):
    start, end = valid
    line = (
        f"{index_key.upper()} 1d: Δ="
        f"{'N/A' if delta_level is None else f'{delta_level:.6f}'} "
        f"(level) A%=N/A (basis n/a valid="
        f"{start.strftime('%Y-%m-%d %H:%M')}->{end.strftime('%Y-%m-%d %H:%M')})"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(line + "\n")

def compute_pct_1d_close(history_csv, index_key):
    """
    history_csv から終値の直近2点を取り、前日比(%)を算出。
    列名は自動検出。prev == 0 の場合は None を返す。
    """
    if not os.path.exists(history_csv):
        return None
    h = pd.read_csv(history_csv)
    if h.empty or len(h) < 2:
        return None
    tcol = guess_time_col(h.columns)
    vcol = guess_value_col(h.drop(columns=[tcol], errors="ignore"), index_key)
    # 日次想定：最後の2点
    try:
        h[tcol] = pd.to_datetime(h[tcol])
    except Exception:
        pass
    h = h.sort_values(tcol)
    last = pd.to_numeric(h[vcol].iloc[-1], errors="coerce")
    prev = pd.to_numeric(h[vcol].iloc[-2], errors="coerce")
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    pct = float((last - prev) / abs(prev) * 100.0)
    return round(pct, 4)

def write_stats_json(path, index_key, delta_level, pct_1d_close):
    payload = {
        "index_key": index_key,
        "pct_1d": None,               # レベル基準の%は常にN/Aポリシー
        "pct_1d_close": pct_1d_close, # X投稿向け（前日終値比, %）
        "delta_level": delta_level,   # 当日始値→終値のレベル差
        "scale": "level",
        "basis": "n/a",
        "updated_at": to_utc_iso(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    index_key = os.environ.get("INDEX_KEY", "ain10")
    out_dir = os.path.join("docs", "outputs")

    # 入力 CSV
    csv_1d = os.path.join(out_dir, f"{index_key}_1d.csv")
    history_csv = os.path.join(out_dir, f"{index_key}_history.csv")

    # 出力
    png_1d = os.path.join(out_dir, f"{index_key}_1d.png")
    post_intraday_txt = os.path.join(out_dir, f"{index_key}_post_intraday.txt")
    stats_json = os.path.join(out_dir, f"{index_key}_stats.json")

    # 1d チャート（黒背景・赤線・%非表示）
    delta_level, valid = render_level_chart(
        csv_1d, png_1d, title=f"{index_key.upper()} (1d)", index_key=index_key
    )

    # X 投稿用の「前日終値比 %」
    pct_1d_close = compute_pct_1d_close(history_csv, index_key)

    # テキストと JSON
    write_post_intraday_txt(post_intraday_txt, index_key, delta_level, valid)
    write_stats_json(stats_json, index_key, delta_level, pct_1d_close)

if __name__ == "__main__":
    main()
