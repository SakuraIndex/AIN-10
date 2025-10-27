# scripts/fallback_intraday.py
# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"

def _auto_datetime_index(df: pd.DataFrame, prefer: str | None) -> pd.DatetimeIndex:
    cand = [prefer] if prefer else []
    cand += list(df.columns)
    for c in cand:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().mean() > 0.6:
                return dt.dt.tz_convert(JST)
    # どうしても無ければ現在時刻を1本だけ
    return pd.DatetimeIndex([pd.Timestamp.now(tz=JST)])

def _is_ratio(values: np.ndarray) -> bool:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return False
    # 95%点が±0.5より小さければ ratio とみなす
    return float(np.quantile(np.abs(values), 0.95)) < 0.5

def main():
    CSV          = Path(os.environ["CSV"])
    OUT_JSON     = Path(os.environ["OUT_JSON"])
    OUT_TEXT     = Path(os.environ["OUT_TEXT"])
    SNAPSHOT_PNG = Path(os.environ["SNAPSHOT_PNG"])
    LABEL        = os.environ.get("LABEL", "AIN-10")
    DT_COL       = os.environ.get("DT_COL", "Datetime")

    df = pd.read_csv(CSV)
    # 日時index付与（自動検出）
    idx = _auto_datetime_index(df, DT_COL if DT_COL in df.columns else None)
    used_col = None
    for c in df.columns:
        try:
            if pd.to_datetime(df[c], errors="coerce").equals(idx.tz_convert(None)):
                used_col = c
                break
        except Exception:
            pass
    df.index = idx
    if used_col and used_col in df.columns:
        df = df.drop(columns=[used_col])

    # 数値列平均 → ％に統一
    num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    s = df[num_cols].mean(axis=1) if num_cols else pd.Series([0.0], index=[pd.Timestamp.now(tz=JST)])
    s = pd.to_numeric(s, errors="coerce")
    arr = s.to_numpy(dtype=float)
    if _is_ratio(arr):
        s = s * 100.0
        arr = s.to_numpy(dtype=float)

    last = float(s.iloc[-1]) if len(s) else 0.0

    # 図（暗色 / プラス青・マイナス赤）
    plt.figure(figsize=(12, 6), dpi=160)
    color = "#00E5FF" if last >= 0 else "#FF4D4D"
    ax = plt.gca()
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")
    plt.plot(s.index, s.values, linewidth=2.0, color=color)
    plt.title(f"{LABEL} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})  {last:+.2f}%")
    plt.xlabel("Time"); plt.ylabel("Change vs Prev Close (%)")
    plt.grid(True, color="#333333", alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    SNAPSHOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SNAPSHOT_PNG, facecolor="#000000")
    plt.close()

    # JSON（ratio 小数で保存）
    payload = {
        "index_key": LABEL,
        "label": LABEL,
        "pct_intraday": round(last / 100.0, 6),
        "basis": "prev_close",
        "session": {},
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # テキスト
    OUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEXT.write_text(
        f"{'▲' if last >= 0 else '▼'} {LABEL} 日中スナップショット ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})\n"
        f"{last:+.2f}%（fallback）\n"
        f"#{LABEL} #日本株",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
