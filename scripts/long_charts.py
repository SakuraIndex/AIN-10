def read_data(path):
    """CSV/TXTから時系列データを抽出（time, value, volume）"""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # --- 🔧 列名自動検出（柔軟化）---
    t_candidates = [
        c for c in df.columns 
        if any(k in c for k in ["time", "date", "datetime", "時刻", "日付"])
    ]
    v_candidates = [
        c for c in df.columns 
        if any(k in c for k in ["close", "price", "value", "index", "終値", "値"])
    ]
    vol_candidates = [
        c for c in df.columns 
        if any(k in c for k in ["volume", "vol", "出来高"])
    ]

    # --- fallback（確実にtcol/vcolを決める）---
    tcol = t_candidates[0] if t_candidates else df.columns[0]
    vcol = v_candidates[0] if v_candidates else df.columns[1]
    volcol = vol_candidates[0] if vol_candidates else None

    def parse_time(x):
        if pd.isna(x): return pd.NaT
        s = str(x)
        if re.fullmatch(r"\d{10}", s):
            return datetime.fromtimestamp(int(s), tz=JST)
        try:
            t = pd.to_datetime(s)
            if t.tzinfo is None:
                t = t.tz_localize(JST)
            return t.tz_convert(JST)
        except Exception:
            return pd.NaT

    df["time"] = df[tcol].apply(parse_time)
    df["value"] = pd.to_numeric(df[vcol], errors="coerce")
    if volcol:
        df["volume"] = pd.to_numeric(df[volcol], errors="coerce")
    else:
        df["volume"] = 0

    df = df.dropna(subset=["time", "value"]).sort_values("time")
    return df[["time", "value", "volume"]]
