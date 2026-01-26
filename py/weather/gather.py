#!/usr/bin/env python3

import csv
from pathlib import Path
import re
import sys

DATE_CSV_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.csv$")

def normalize_header(header):
    """
    Normalize header fields so cosmetic differences don't break matching.
    - strip BOM (handled by utf-8-sig)
    - strip whitespace
    - strip surrounding quotes
    """
    return [h.strip().strip('"') for h in header]

def build_year(year_dir: Path):
    year = year_dir.name
    output_file = year_dir / f"{year}_BCWS_WX_OBS.csv"

    if output_file.exists():
        print(f"[SKIP] {year}: annual WX_OBS already exists")
        return

    daily_files = sorted(
        f for f in year_dir.iterdir()
        if f.is_file() and DATE_CSV_RE.match(f.name)
    )

    if not daily_files:
        print(f"[SKIP] {year}: no daily CSV files found")
        return

    print(f"[BUILD] {year}: {len(daily_files)} daily files")

    reference_header = None
    rows_written = 0

    with output_file.open("w", newline="", encoding="utf-8") as out_fp:
        writer = None

        for daily in daily_files:
            with daily.open("r", newline="", encoding="utf-8-sig") as in_fp:
                reader = csv.reader(in_fp)

                try:
                    header = next(reader)
                except StopIteration:
                    continue  # empty file

                header = normalize_header(header)

                if reference_header is None:
                    reference_header = header
                    writer = csv.writer(out_fp)
                    writer.writerow(reference_header)
                else:
                    if header != reference_header:
                        raise ValueError(
                            f"Header mismatch in {daily}\n"
                            f"Expected: {reference_header}\n"
                            f"Found:    {header}"
                        )

                for row in reader:
                    if not row or all(not cell.strip() for cell in row):
                        continue
                    writer.writerow([cell.strip() for cell in row])
                    rows_written += 1

    print(f"[DONE ] {year}: wrote {rows_written} rows → {output_file.name}")

def main(root):
    root = Path(root)

    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    for year_dir in sorted(root.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            build_year(year_dir)

if __name__ == "__main__":
    main("./BCWS_DATA_MART")



