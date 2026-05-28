from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .download_announcements import (
    DEFAULT_STOCK_CODES,
    collect_stock_codes,
    normalize_announcements,
    _load_cninfo_raw,
)


OUTPUT_COLUMNS = ["stock_code", "trade_date", "text", "source", "url"]
EASTMONEY_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"


def _empty_text_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _strip_html(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"</?em>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {str(col).strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _extract_jsonp_payload(text: str) -> dict:
    start = text.find("(")
    end = text.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("EastMoney response is not valid JSONP.")
    return json.loads(text[start + 1 : end])


def _load_eastmoney_news_page(symbol: str, page_index: int, page_size: int) -> pd.DataFrame:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Install `requests` before downloading EastMoney stock news.") from exc

    inner_param = {
        "uid": "",
        "keyword": symbol,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": page_index,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    callback = f"jQuery_stock_news_{int(time.time() * 1000)}"
    params = {
        "cb": callback,
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Referer": f"https://so.eastmoney.com/news/s?keyword={symbol}",
    }
    response = requests.get(EASTMONEY_SEARCH_URL, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    payload = _extract_jsonp_payload(response.text)
    rows = payload.get("result", {}).get("cmsArticleWebOld", [])
    return pd.DataFrame(rows)


def _load_eastmoney_stock_news_raw(
    stock_code: str,
    max_rows: int | None = None,
    page_size: int = 50,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    symbol = stock_code.split(".")[0]
    frames: list[pd.DataFrame] = []
    target_rows = max_rows or 100
    max_pages = max(1, (target_rows + page_size - 1) // page_size)

    direct_error: Exception | None = None
    try:
        for page_index in range(1, max_pages + 1):
            one = _load_eastmoney_news_page(symbol, page_index=page_index, page_size=page_size)
            if one.empty:
                break
            frames.append(one)
            if sum(len(frame) for frame in frames) >= target_rows:
                break
            time.sleep(sleep_seconds)
    except Exception as exc:  # pragma: no cover - depends on remote endpoint stability
        direct_error = exc

    if frames:
        return pd.concat(frames, ignore_index=True)

    try:
        import akshare as ak  # type: ignore

        return ak.stock_news_em(symbol=symbol)
    except Exception as exc:  # pragma: no cover - depends on remote endpoint stability
        if direct_error is not None:
            raise RuntimeError(f"EastMoney direct API failed: {direct_error}; AkShare fallback failed: {exc}") from exc
        raise


def normalize_stock_news(raw: pd.DataFrame, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return _empty_text_frame()

    title_col = _pick_column(raw, ["新闻标题", "title", "Title"])
    summary_col = _pick_column(raw, ["新闻内容", "content", "summary", "摘要"])
    date_col = _pick_column(raw, ["发布时间", "date", "publish_time", "发布时间 "])
    url_col = _pick_column(raw, ["新闻链接", "url", "Url"])

    if title_col is None:
        raise ValueError(f"Cannot find stock news title column. Available columns: {list(raw.columns)}")
    if date_col is None:
        raise ValueError(f"Cannot find stock news date column. Available columns: {list(raw.columns)}")

    out = pd.DataFrame()
    titles = raw[title_col].map(_strip_html)
    if summary_col:
        summaries = raw[summary_col].map(_strip_html)
        out["text"] = [
            f"{title}。{summary}" if summary and summary not in title else title
            for title, summary in zip(titles, summaries)
        ]
    else:
        out["text"] = titles

    out["stock_code"] = stock_code
    out["trade_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["source"] = "eastmoney_stock_news"
    out["url"] = raw[url_col].fillna("").astype(str) if url_col else ""

    out = out.dropna(subset=["trade_date"])
    out = out[out["text"].astype(str).str.strip() != ""].copy()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dates = pd.to_datetime(out["trade_date"])
    out = out[(dates >= start_ts) & (dates <= end_ts)].copy()
    return out[OUTPUT_COLUMNS].drop_duplicates()


def collect_stock_texts(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    output: str | Path,
    sources: list[str] | None = None,
    max_rows_per_stock: int | None = None,
    sleep_seconds: float = 0.5,
    eastmoney_page_size: int = 50,
) -> pd.DataFrame:
    sources = sources or ["cninfo", "eastmoney"]
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str, str]] = []

    for stock_code in stock_codes:
        stock_frames: list[pd.DataFrame] = []
        if "cninfo" in sources:
            try:
                raw = _load_cninfo_raw(stock_code, start_date, end_date)
                one = normalize_announcements(raw, stock_code, start_date, end_date)
                stock_frames.append(one)
                print(f"[OK] {stock_code} cninfo: {len(one)} rows")
            except Exception as exc:  # pragma: no cover - network/API failures are environment specific
                failures.append((stock_code, "cninfo", str(exc)))
                print(f"[FAIL] {stock_code} cninfo: {exc}")

        if "eastmoney" in sources:
            try:
                raw = _load_eastmoney_stock_news_raw(
                    stock_code,
                    max_rows=max_rows_per_stock,
                    page_size=eastmoney_page_size,
                    sleep_seconds=sleep_seconds,
                )
                one = normalize_stock_news(raw, stock_code, start_date, end_date)
                stock_frames.append(one)
                print(f"[OK] {stock_code} eastmoney: {len(one)} rows")
            except Exception as exc:  # pragma: no cover - network/API failures are environment specific
                failures.append((stock_code, "eastmoney", str(exc)))
                print(f"[FAIL] {stock_code} eastmoney: {exc}")

        if stock_frames:
            stock_result = pd.concat(stock_frames, ignore_index=True)
            stock_result = stock_result.drop_duplicates(subset=["stock_code", "trade_date", "text", "source"])
            if max_rows_per_stock:
                stock_result = stock_result.sort_values(["trade_date", "source"]).tail(max_rows_per_stock)
            frames.append(stock_result)
        time.sleep(sleep_seconds)

    result = pd.concat(frames, ignore_index=True) if frames else _empty_text_frame()
    result = result.sort_values(["stock_code", "trade_date", "source"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    if failures:
        fail_path = output.with_suffix(".failures.csv")
        pd.DataFrame(failures, columns=["stock_code", "source", "error"]).to_csv(
            fail_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[WARN] failures saved to {fail_path}")

    print(f"[DONE] stock text saved to {output}, rows={len(result)}")
    return result


def disable_proxy_env() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download stock announcements and news into data/raw/news.csv.")
    parser.add_argument("--stock-code", action="append", help="Stock code, e.g. 000001.SZ. Can be used repeatedly.")
    parser.add_argument("--stock-list", help="TXT/CSV file containing stock codes. CSV may contain a stock_code column.")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date, e.g. 2023-01-01.")
    parser.add_argument("--end-date", default="2024-12-31", help="End date, e.g. 2024-12-31.")
    parser.add_argument("--output", default="data/raw/news.csv", help="Output CSV path.")
    parser.add_argument(
        "--source",
        action="append",
        choices=["cninfo", "eastmoney"],
        help="Text source to download. Can be used repeatedly. Defaults to cninfo + eastmoney.",
    )
    parser.add_argument("--max-rows-per-stock", type=int, help="Keep the latest N text rows per stock.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between requests/stocks.")
    parser.add_argument("--eastmoney-page-size", type=int, default=50, help="EastMoney search page size.")
    parser.add_argument("--no-proxy", action="store_true", help="Clear proxy environment variables before downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_proxy:
        disable_proxy_env()
    stock_codes = collect_stock_codes(args.stock_code, args.stock_list)
    if not stock_codes:
        stock_codes = DEFAULT_STOCK_CODES
    print(f"[INFO] downloading stock texts for {len(stock_codes)} stocks: {', '.join(stock_codes)}")
    collect_stock_texts(
        stock_codes=stock_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        sources=args.source,
        max_rows_per_stock=args.max_rows_per_stock,
        sleep_seconds=args.sleep,
        eastmoney_page_size=args.eastmoney_page_size,
    )


if __name__ == "__main__":
    main()
