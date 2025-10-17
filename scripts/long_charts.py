#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("docs/outputs")
OUT.mkdir(parents=True, exist_ok=True)
INDEX_KEY = "ain10"

def _load(name):
    path = OUT / f"{INDEX_KEY}_{name}.csv"
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Invalid CSV: {path}")
    ts_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[ts_col, val_col]].rename(columns={ts_col: "time", val_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df

def _plot(df, title, out_png):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10,5), dpi=130)
    ax.plot(df["time"], df["value"], color="red", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index (level)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    for span in ["1d","7d","1m","1y"]:
        try:
            df = _load(span)
            _plot(df, f"{INDEX_KEY.upper()} ({span})", OUT / f"{INDEX_KEY}_{span}.png")
        except Exception as e:
            print(f"[WARN] skip {span}: {e}")

if __name__ == "__main__":
    main()
