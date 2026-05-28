from __future__ import annotations

import argparse
from bisect import bisect_left
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["stock_code", "trade_date", "text", "source", "url"]


def align_text_dates_to_market(
    text_csv: str | Path,
    market_csv: str | Path,
    output: str | Path,
    drop_unmatched: bool = True,
) -> pd.DataFrame:
    text = pd.read_csv(text_csv)
    market = pd.read_csv(market_csv)
    missing = set(OUTPUT_COLUMNS) - set(text.columns)
    if missing:
        raise ValueError(f"text csv missing columns: {sorted(missing)}")
    for col in ["stock_code", "trade_date"]:
        if col not in market.columns:
            raise ValueError(f"market csv missing required column: {col}")

    text = text[OUTPUT_COLUMNS].copy()
    text["stock_code"] = text["stock_code"].astype(str)
    text["trade_date"] = pd.to_datetime(text["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    market["stock_code"] = market["stock_code"].astype(str)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    calendars = {
        stock_code: sorted(values.dropna().astype(str).unique().tolist())
        for stock_code, values in market.groupby("stock_code")["trade_date"]
    }

    def map_date(row: pd.Series) -> str | None:
        dates = calendars.get(str(row["stock_code"]))
        if not dates:
            return None
        date = str(row["trade_date"])
        idx = bisect_left(dates, date)
        if idx >= len(dates):
            return None
        return dates[idx]

    text["trade_date"] = text.apply(map_date, axis=1)
    if drop_unmatched:
        text = text.dropna(subset=["trade_date"]).copy()
    text = text.drop_duplicates(subset=["stock_code", "trade_date", "text", "source"])
    text = text.sort_values(["stock_code", "trade_date", "source"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[DONE] aligned text dates to {output}, rows={len(text)}")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map stock text dates to the next available market trade_date.")
    parser.add_argument("--text-csv", required=True)
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-unmatched", action="store_true", help="Keep rows that cannot be mapped to a market date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    align_text_dates_to_market(
        text_csv=args.text_csv,
        market_csv=args.market_csv,
        output=args.output,
        drop_unmatched=not args.keep_unmatched,
    )


if __name__ == "__main__":
    main()
