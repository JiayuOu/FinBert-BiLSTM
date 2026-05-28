from __future__ import annotations

from pathlib import Path
import pandas as pd

from .features import normalize_market_columns


def load_market_csv(path: str | Path) -> pd.DataFrame:
    return normalize_market_columns(pd.read_csv(path))


def fetch_akshare_daily(stock_code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch A-share daily data with Akshare if the optional dependency exists."""
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise ImportError("Install optional dependency `akshare` to use this data source.") from exc

    symbol = stock_code.split(".")[0]
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    try:
        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
    except Exception as em_exc:
        tx_symbol = _to_tencent_symbol(stock_code)
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol,
                start_date=start,
                end_date=end,
                adjust=adjust or "",
            )
            print(f"[WARN] Eastmoney failed for {stock_code}; used Tencent fallback. Error: {em_exc}")
        except Exception as tx_exc:
            raise RuntimeError(f"Eastmoney failed: {em_exc}; Tencent fallback failed: {tx_exc}") from tx_exc
    raw["stock_code"] = stock_code
    return normalize_market_columns(raw)


def _to_tencent_symbol(stock_code: str) -> str:
    code = stock_code.split(".")[0]
    suffix = stock_code.split(".")[-1].upper() if "." in stock_code else ""
    if suffix == "SH" or code.startswith("6"):
        return f"sh{code}"
    return f"sz{code}"


def fetch_tushare_daily(stock_code: str, start_date: str, end_date: str, token: str | None = None) -> pd.DataFrame:
    """Fetch A-share daily data with Tushare if the optional dependency exists."""
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:
        raise ImportError("Install optional dependency `tushare` to use this data source.") from exc

    if token:
        ts.set_token(token)
    pro = ts.pro_api()
    raw = pro.daily(ts_code=stock_code, start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""))
    return normalize_market_columns(raw)
