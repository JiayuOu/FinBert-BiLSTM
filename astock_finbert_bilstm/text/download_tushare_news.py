from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


TUSHARE_API_URL = "http://api.tushare.pro"
OUTPUT_COLUMNS = ["stock_code", "trade_date", "text", "source", "url"]
DEFAULT_SOURCES = ["sina", "eastmoney", "10jqka", "cls", "yicai"]


def _clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _load_stock_pool(path: str | Path) -> pd.DataFrame:
    pool = pd.read_csv(path)
    if "stock_code" not in pool.columns:
        pool = pool.rename(columns={pool.columns[0]: "stock_code"})
    if "name" not in pool.columns:
        pool["name"] = ""
    pool["stock_code"] = pool["stock_code"].astype(str)
    pool["code6"] = pool["stock_code"].str.split(".").str[0].str.zfill(6)
    pool["name"] = pool["name"].fillna("").astype(str).str.strip()
    return pool[["stock_code", "code6", "name"]].drop_duplicates()


def _date_windows(start_date: str, end_date: str, days: int) -> list[tuple[datetime, datetime]]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    windows = []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=days) - timedelta(seconds=1), end)
        windows.append((current, window_end))
        current = window_end + timedelta(seconds=1)
    return windows


def _call_tushare_news(token: str, source: str, start: datetime, end: datetime) -> pd.DataFrame:
    payload = {
        "api_name": "news",
        "token": token,
        "params": {
            "src": source,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "fields": "datetime,title,content,channels",
    }
    response = requests.post(TUSHARE_API_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Tushare error code={data.get('code')}: {data.get('msg')}")
    fields = data.get("data", {}).get("fields", [])
    items = data.get("data", {}).get("items", [])
    return pd.DataFrame(items, columns=fields)


def _match_news_to_stocks(raw: pd.DataFrame, pool: pd.DataFrame, source: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict[str, str]] = []
    for item in raw.itertuples(index=False):
        title = _clean_text(getattr(item, "title", ""))
        content = _clean_text(getattr(item, "content", ""))
        dt = getattr(item, "datetime", "")
        if not title and not content:
            continue
        haystack = f"{title} {content}"
        text = title if not content or content in title else f"{title}。{content}"
        trade_date = pd.to_datetime(dt, errors="coerce")
        if pd.isna(trade_date):
            continue
        for stock in pool.itertuples(index=False):
            aliases = [str(stock.name).strip(), str(stock.code6).strip()]
            if any(alias and alias in haystack for alias in aliases):
                rows.append(
                    {
                        "stock_code": str(stock.stock_code),
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "text": text,
                        "source": f"tushare_{source}_news",
                        "url": "",
                    }
                )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).drop_duplicates()


def download_tushare_news(
    stock_list: str | Path,
    start_date: str,
    end_date: str,
    output: str | Path,
    token: str,
    sources: list[str] | None = None,
    window_days: int = 7,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    pool = _load_stock_pool(stock_list)
    sources = sources or DEFAULT_SOURCES
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str, str]] = []

    for source in sources:
        for start, end in _date_windows(start_date, end_date, window_days):
            try:
                raw = _call_tushare_news(token, source, start, end)
                matched = _match_news_to_stocks(raw, pool, source)
                if not matched.empty:
                    frames.append(matched)
                print(f"[OK] {source} {start:%Y-%m-%d}..{end:%Y-%m-%d}: raw={len(raw)} matched={len(matched)}")
            except Exception as exc:  # pragma: no cover - remote API and permissions are environment specific
                failures.append((source, f"{start:%Y-%m-%d}..{end:%Y-%m-%d}", str(exc)))
                print(f"[FAIL] {source} {start:%Y-%m-%d}..{end:%Y-%m-%d}: {exc}")
            time.sleep(sleep_seconds)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = result.drop_duplicates(subset=["stock_code", "trade_date", "text", "source"])
    result = result.sort_values(["stock_code", "trade_date", "source"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    if failures:
        fail_path = output.with_suffix(".failures.csv")
        pd.DataFrame(failures, columns=["source", "window", "error"]).to_csv(fail_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] failures saved to {fail_path}")
    print(f"[DONE] tushare matched news saved to {output}, rows={len(result)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tushare news and match it to a stock pool.")
    parser.add_argument("--stock-list", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token", default=os.environ.get("TUSHARE_TOKEN"), help="Tushare token, or set TUSHARE_TOKEN.")
    parser.add_argument("--source", action="append", choices=DEFAULT_SOURCES, help="News source. Repeat to use multiple.")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Tushare token. Pass --token or set TUSHARE_TOKEN.")
    download_tushare_news(
        stock_list=args.stock_list,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        token=args.token,
        sources=args.source,
        window_days=args.window_days,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
