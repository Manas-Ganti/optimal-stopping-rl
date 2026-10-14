"""Daily prices from free sources, cached to disk.

Uses the *split-adjusted, not dividend-adjusted* close (yfinance `auto_adjust=False`, column
`Close`): the option is written on the traded price, which does drop on ex-dividend dates.
Falls back to Stooq if yfinance fails (Stooq's series are also adjusted for dividends, which
the cache records as `source`).

Real data has exactly three uses in this project: calibrating vol, out-of-sample evaluation,
and legibility. It is never used for training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optstop.config import REPO_ROOT


def fetch_close(ticker: str, start: str, end: str, cache_dir: str = "data_cache") -> pd.Series:
    cache = REPO_ROOT / cache_dir / f"{ticker}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df["close"].rename(ticker)
    cache.parent.mkdir(parents=True, exist_ok=True)
    s, source = None, None
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False,
                         threads=False)
        if df is not None and len(df):
            close = df["Close"]
            s = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()
            source = "yfinance"
    except Exception as e:  # network / API changes
        print(f"  yfinance failed for {ticker}: {e!r}; trying Stooq")
    if s is None or not len(s):
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = df["Close"].loc[start:end].dropna()
        source = "stooq"
    if not len(s):
        raise RuntimeError(f"no data for {ticker}")
    out = pd.DataFrame({"close": s.astype(float), "source": source})
    out.index.name = "date"
    out.to_csv(cache)
    return out["close"].rename(ticker)
