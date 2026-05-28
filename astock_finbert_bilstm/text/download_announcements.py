from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


DEFAULT_STOCK_CODES = [
    "000001.SZ",
    "000002.SZ",
    "600000.SH",
    "600519.SH",
    "000858.SZ",
]


COLUMN_CANDIDATES = {
    "stock_code": ["stock_code", "证券代码", "代码", "股票代码"],
    "trade_date": ["trade_date", "公告时间", "公告日期", "日期", "发布时间"],
    "title": ["title", "公告标题", "标题", "announcement_title"],
    "url": ["url", "公告链接", "链接", "announcement_url"],
}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_stock_codes(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "stock_code" in df.columns:
            values = df["stock_code"].dropna().astype(str).tolist()
        else:
            values = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value and not value.startswith("#")]


def collect_stock_codes(cli_codes: list[str] | None, stock_list: str | Path | None) -> list[str]:
    codes = []
    codes.extend(cli_codes or [])
    codes.extend(read_stock_codes(stock_list))
    if not codes:
        codes = DEFAULT_STOCK_CODES
    return list(dict.fromkeys(code.strip() for code in codes if code.strip()))


def _load_cninfo_raw(
    stock_code: str,
    start_date: str,
    end_date: str,
    keyword: str = "",
    category: str = "",
) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise ImportError("Install `akshare` before downloading announcements: pip install akshare") from exc

    symbol = stock_code.split(".")[0]
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    if hasattr(ak, "stock_zh_a_disclosure_report_cninfo"):
        return ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol,
            market="沪深京",
            keyword=keyword,
            category=category,
            start_date=start,
            end_date=end,
        )
    if hasattr(ak, "stock_notice_report"):
        return ak.stock_notice_report(symbol=symbol)
    raise RuntimeError("Current akshare version does not provide a supported CNINFO announcement API.")


def normalize_announcements(raw: pd.DataFrame, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["stock_code", "trade_date", "text", "source", "url"])

    code_col = _pick_column(raw, COLUMN_CANDIDATES["stock_code"])
    date_col = _pick_column(raw, COLUMN_CANDIDATES["trade_date"])
    title_col = _pick_column(raw, COLUMN_CANDIDATES["title"])
    url_col = _pick_column(raw, COLUMN_CANDIDATES["url"])

    if title_col is None:
        raise ValueError(f"Cannot find announcement title column. Available columns: {list(raw.columns)}")
    if date_col is None:
        raise ValueError(f"Cannot find announcement date column. Available columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out["stock_code"] = raw[code_col].astype(str) if code_col else stock_code
    out["trade_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["text"] = raw[title_col].fillna("").astype(str).str.strip()
    out["source"] = "cninfo_announcement_title"
    out["url"] = raw[url_col].fillna("").astype(str) if url_col else ""

    out = out.dropna(subset=["trade_date"])
    out = out[out["text"] != ""].copy()
    out["stock_code"] = stock_code

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dates = pd.to_datetime(out["trade_date"])
    out = out[(dates >= start_ts) & (dates <= end_ts)].copy()

    return out[["stock_code", "trade_date", "text", "source", "url"]].drop_duplicates()


def download_announcements(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    output: str | Path,
    keyword: list[str] | None = None,
    category: str = "",
    max_rows_per_stock: int | None = None,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    keyword = keyword or []
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for stock_code in stock_codes:
        try:
            api_keyword = keyword[0] if keyword else ""
            raw = _load_cninfo_raw(stock_code, start_date, end_date, keyword=api_keyword, category=category)
            one = normalize_announcements(raw, stock_code, start_date, end_date)
            if keyword:
                pattern = "|".join(keyword)
                one = one[one["text"].str.contains(pattern, case=False, na=False, regex=True)].copy()
            if max_rows_per_stock:
                one = one.sort_values("trade_date").tail(max_rows_per_stock)
            frames.append(one)
            print(f"[OK] {stock_code}: {len(one)} rows")
        except Exception as exc:  # pragma: no cover - network/API failures are environment specific
            failures.append((stock_code, str(exc)))
            print(f"[FAIL] {stock_code}: {exc}")
        time.sleep(sleep_seconds)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["stock_code", "trade_date", "text", "source", "url"])
    result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    if failures:
        fail_path = output.with_suffix(".failures.csv")
        pd.DataFrame(failures, columns=["stock_code", "error"]).to_csv(fail_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] failures saved to {fail_path}")

    print(f"[DONE] announcement text saved to {output}, rows={len(result)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CNINFO announcements into data/raw/news.csv.")
    parser.add_argument("--stock-code", action="append", help="Stock code, e.g. 000001.SZ. Can be used repeatedly.")
    parser.add_argument("--stock-list", help="TXT/CSV file containing stock codes. CSV may contain a stock_code column.")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date, e.g. 2023-01-01.")
    parser.add_argument("--end-date", default="2024-12-31", help="End date, e.g. 2024-12-31.")
    parser.add_argument("--output", default="data/raw/news.csv", help="Output CSV path.")
    parser.add_argument("--keyword", action="append", help="Keep only rows whose title contains this keyword.")
    parser.add_argument("--category", default="", help="CNINFO category, e.g. 年报, 业绩预告, 日常经营, 风险提示.")
    parser.add_argument("--max-rows-per-stock", type=int, help="Keep the latest N rows per stock after filtering.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between stocks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_codes = collect_stock_codes(args.stock_code, args.stock_list)
    print(f"[INFO] downloading announcements for {len(stock_codes)} stocks: {', '.join(stock_codes)}")
    download_announcements(
        stock_codes=stock_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        keyword=args.keyword,
        category=args.category,
        max_rows_per_stock=args.max_rows_per_stock,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
