#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--title", default="Intraday (fallback)")
    ap.add_argument("--dt-col", default="Datetime")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 数値列の平均で一本線（Datetime は除外）
    num_cols = [c for c in df.columns if c != args.dt_col]
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        y = df[num_cols].mean(axis=1).to_list()
        x = range(len(y))
    else:
        x = [0]
        y = [0]

    plt.figure(figsize=(8, 3), dpi=160)
    plt.plot(list(x), list(y))
    plt.title(args.title)
    plt.xlabel("Time")
    plt.ylabel("Index (level)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
