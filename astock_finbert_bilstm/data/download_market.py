from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from .features import normalize_market_columns
from .sources import fetch_akshare_daily


DEFAULT_STOCK_CODES = [
    "000001.SZ",
    "000002.SZ",
    "600000.SH",
    "600519.SH",
    "000858.SZ",
]


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


def download_market_data(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    output: str | Path,
    adjust: str = "qfq",
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for stock_code in stock_codes:
        try:
            one = fetch_akshare_daily(stock_code, start_date, end_date, adjust=adjust)
            source_code_columns = [col for col in ["stock_code", "股票代码", "代码", "ts_code"] if col in one.columns]
            if source_code_columns:
                one = one.drop(columns=source_code_columns)
            one["stock_code"] = stock_code
            frames.append(one)
            print(f"[OK] {stock_code}: {len(one)} rows")
        except Exception as exc:  # pragma: no cover - network/API failures are environment specific
            failures.append((stock_code, str(exc)))
            print(f"[FAIL] {stock_code}: {exc}")
        time.sleep(sleep_seconds)

    if not frames:
        detail = "; ".join(f"{code}: {message}" for code, message in failures)
        raise RuntimeError(f"No market data downloaded. {detail}")

    result = normalize_market_columns(pd.concat(frames, ignore_index=True))
    keep_first = [
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "pct_chg",
    ]
    ordered = [col for col in keep_first if col in result.columns]
    ordered += [col for col in result.columns if col not in ordered]
    result = result[ordered].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    if failures:
        fail_path = output.with_suffix(".failures.csv")
        pd.DataFrame(failures, columns=["stock_code", "error"]).to_csv(fail_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] failures saved to {fail_path}")

    print(f"[DONE] market data saved to {output}, rows={len(result)}")
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
    parser = argparse.ArgumentParser(description="Download A-share daily market data into data/raw/market.csv.")
    parser.add_argument("--stock-code", action="append", help="Stock code, e.g. 000001.SZ. Can be used repeatedly.")
    parser.add_argument("--stock-list", help="TXT/CSV file containing stock codes. CSV may contain a stock_code column.")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date, e.g. 2023-01-01.")
    parser.add_argument("--end-date", default="2024-12-31", help="End date, e.g. 2024-12-31.")
    parser.add_argument("--output", default="data/raw/market.csv", help="Output CSV path.")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="Akshare adjustment mode.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between stocks.")
    parser.add_argument("--no-proxy", action="store_true", help="Clear proxy environment variables before downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_proxy:
        disable_proxy_env()
    stock_codes = collect_stock_codes(args.stock_code, args.stock_list)
    print(f"[INFO] downloading {len(stock_codes)} stocks: {', '.join(stock_codes)}")
    download_market_data(
        stock_codes=stock_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        adjust=args.adjust,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
