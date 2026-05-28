from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd



OUTPUT_COLUMNS = ["stock_code", "trade_date", "text", "source", "url"]


def _normalize_code(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    raw = re.sub(r"\.0$", "", raw)
    if "." in raw and raw.split(".")[-1].upper() in {"SZ", "SH", "BJ"}:
        code, suffix = raw.split(".", 1)
        return f"{code.zfill(6)}.{suffix.upper()}"
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return raw
    code = digits.zfill(6)
    suffix = "SH" if code.startswith(("6", "9")) else "BJ" if code.startswith(("4", "8")) else "SZ"
    return f"{code}.{suffix}"


def _clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _load_stock_pool(path: str | Path) -> pd.DataFrame:
    pool = pd.read_csv(path)
    if "stock_code" not in pool.columns:
        first = pool.columns[0]
        pool = pool.rename(columns={first: "stock_code"})
    pool["stock_code"] = pool["stock_code"].map(_normalize_code)
    if "name" not in pool.columns:
        pool["name"] = ""
    return pool[["stock_code", "name"]].drop_duplicates()


def import_astock_csv(
    input_path: str | Path,
    stock_list: str | Path,
    output: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    source_name: str = "astock_stock_news",
) -> pd.DataFrame:
    pool = _load_stock_pool(stock_list)
    pool_codes = set(pool["stock_code"].astype(str))
    name_to_code = {
        str(row.name).strip(): str(row.stock_code)
        for row in pool.itertuples(index=False)
        if str(row.name).strip()
    }

    raw = pd.read_csv(input_path, sep="\t", dtype=str)
    if "CODE" not in raw.columns and "stock_code" not in raw.columns:
        raise ValueError(f"Astock input missing CODE column: {list(raw.columns)}")
    if "DATE" not in raw.columns and "trade_date" not in raw.columns:
        raise ValueError(f"Astock input missing DATE column: {list(raw.columns)}")

    code_col = "CODE" if "CODE" in raw.columns else "stock_code"
    date_col = "DATE" if "DATE" in raw.columns else "trade_date"
    text_col = "text_a" if "text_a" in raw.columns else "DESCRIPTION" if "DESCRIPTION" in raw.columns else None
    if text_col is None:
        raise ValueError(f"Astock input missing text_a/DESCRIPTION column: {list(raw.columns)}")

    out = pd.DataFrame()
    out["stock_code"] = raw[code_col].map(_normalize_code)
    if "NAME" in raw.columns:
        names = raw["NAME"].fillna("").astype(str).str.strip()
        out.loc[~out["stock_code"].isin(pool_codes), "stock_code"] = names.map(name_to_code)

    out["trade_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["text"] = raw[text_col].map(_clean_text)
    out["source"] = source_name
    out["url"] = ""

    out = out[out["stock_code"].isin(pool_codes)].copy()
    out = out.dropna(subset=["stock_code", "trade_date"])
    out = out[out["text"] != ""].copy()
    if start_date:
        out = out[pd.to_datetime(out["trade_date"]) >= pd.Timestamp(start_date)].copy()
    if end_date:
        out = out[pd.to_datetime(out["trade_date"]) <= pd.Timestamp(end_date)].copy()

    out = out[OUTPUT_COLUMNS].drop_duplicates()
    out = out.sort_values(["stock_code", "trade_date", "text"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[DONE] imported {len(out)} rows from {input_path} to {output}")
    return out


def merge_text_csvs(inputs: list[str | Path], output: str | Path) -> pd.DataFrame:
    frames = []
    for path in inputs:
        one = pd.read_csv(path)
        missing = set(OUTPUT_COLUMNS) - set(one.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(one[OUTPUT_COLUMNS])
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result["stock_code"] = result["stock_code"].astype(str).map(_normalize_code)
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["text"] = result["text"].map(_clean_text)
    result = result.dropna(subset=["stock_code", "trade_date"])
    result = result[result["text"] != ""].copy()
    result = result.drop_duplicates(subset=["stock_code", "trade_date", "text", "source"])
    result = result.sort_values(["stock_code", "trade_date", "source"]).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[DONE] merged {len(inputs)} files into {output}, rows={len(result)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import or merge external stock texts into news.csv-compatible format.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    astock = subparsers.add_parser("astock", help="Import Astock TSV/CSV data.")
    astock.add_argument("--input", required=True, help="Astock TSV file, e.g. data/df_all_year_srl.csv.")
    astock.add_argument("--stock-list", required=True, help="CSV stock pool with stock_code and name columns.")
    astock.add_argument("--output", required=True)
    astock.add_argument("--start-date")
    astock.add_argument("--end-date")
    astock.add_argument("--source-name", default="astock_stock_news")

    merge = subparsers.add_parser("merge", help="Merge news.csv-compatible files.")
    merge.add_argument("--input", action="append", required=True, help="Input CSV. Can be used repeatedly.")
    merge.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "astock":
        import_astock_csv(
            input_path=args.input,
            stock_list=args.stock_list,
            output=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            source_name=args.source_name,
        )
    elif args.command == "merge":
        merge_text_csvs(args.input, args.output)


if __name__ == "__main__":
    main()
