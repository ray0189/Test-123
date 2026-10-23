"""Download Dukascopy XAUUSD M1 bid/ask candles and build a parquet file.

Usage:
    python scripts/download_dukascopy.py --start 2022-01-01 --end 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from raygold.data.dukascopy import build_m1, download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--cache", default="data/raw/dukascopy")
    ap.add_argument("--out", default="data/XAUUSD_M1_bidask.parquet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    cache = Path(args.cache)
    failed = download(args.symbol, start, end, cache, workers=args.workers)
    if failed:
        logging.warning("retrying %d failed files sequentially", len(failed))
        failed = download(args.symbol, start, end, cache, workers=1)
    if failed:
        logging.error("still missing %d day-files: %s", len(failed), failed[:20])
    df = build_m1(args.symbol, start, end, cache)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logging.info("wrote %d M1 rows %s .. %s to %s", len(df), df["time"].iloc[0], df["time"].iloc[-1], args.out)


if __name__ == "__main__":
    main()
