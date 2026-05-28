from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Chinese/Akshare/Tushare column names to internal names."""
    rename_map = {
        "日期": "trade_date",
        "date": "trade_date",
        "Date": "trade_date",
        "股票代码": "stock_code",
        "代码": "stock_code",
        "symbol": "stock_code",
        "Symbol": "stock_code",
        "开盘": "open",
        "open": "open",
        "Open": "open",
        "最高": "high",
        "high": "high",
        "High": "high",
        "最低": "low",
        "low": "low",
        "Low": "low",
        "收盘": "close",
        "close": "close",
        "Close": "close",
        "成交量": "volume",
        "amount": "amount",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "涨跌幅": "pct_chg",
        "ts_code": "stock_code",
        "trade_date": "trade_date",
        "vol": "volume",
        "volume": "volume",
        "Volume": "volume",
    }
    out = df.rename(columns={c: rename_map.get(c, c) for c in df.columns}).copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    if "stock_code" in out.columns:
        out["stock_code"] = out["stock_code"].astype(str)
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple thesis-friendly indicators per stock without future leakage."""
    required = {"stock_code", "trade_date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"market data missing required columns: {sorted(missing)}")

    out = normalize_market_columns(df)
    out = out.sort_values(["stock_code", "trade_date"]).copy()

    def apply_one(stock: pd.DataFrame) -> pd.DataFrame:
        stock = stock.copy()
        close = stock["close"].astype(float)

        stock["ma5"] = close.rolling(5, min_periods=1).mean()
        stock["ma10"] = close.rolling(10, min_periods=1).mean()
        stock["ma20"] = close.rolling(20, min_periods=1).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        stock["rsi14"] = (100 - 100 / (1 + rs)).fillna(50.0)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        stock["macd"] = ema12 - ema26
        stock["macd_signal"] = stock["macd"].ewm(span=9, adjust=False).mean()
        stock["macd_hist"] = stock["macd"] - stock["macd_signal"]

        if "pct_chg" not in stock.columns:
            stock["pct_chg"] = close.pct_change().fillna(0.0) * 100
        if "turnover_rate" not in stock.columns:
            stock["turnover_rate"] = 0.0
        if "volume" not in stock.columns:
            stock["volume"] = 0.0
        return stock

    frames = [apply_one(stock) for _, stock in out.groupby("stock_code", sort=False)]
    return pd.concat(frames, ignore_index=True)
