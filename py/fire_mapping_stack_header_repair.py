'''20260622: stack file's header gets messed up sometimes.

terminate the band names field properly, and restore map / proj / CRS info from the corresponding post MRAP file.
'''
#!/usr/bin/env python3
import sys
import os
import re
from pathlib import Path


REMOVE_KEYS = {
    "map info",
    "projection info",
    "coordinate system string",
}


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def backup_once(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as src, open(bak, "wb") as dst:
            dst.write(src.read())


def split_records(text):
    """
    Split ENVI header into {key: value} records.
    Handles multiline {...} blocks safely.
    """
    lines = text.splitlines()
    records = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if "=" not in line:
            i += 1
            continue

        key, val = line.split("=", 1)
        key = key.strip().lower()
        val = val.strip()

        # multiline brace block
        if "{" in val and "}" not in val:
            buf = [val]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if "}" in lines[i]:
                    break
                i += 1
            val = "\n".join(buf)

        records.append((key, val))
        i += 1

    return records


def parse_date_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"(\d{8})_stack\.hdr", name)
    if not m:
        raise ValueError("Filename must be yyyymmdd_stack.hdr")
    return m.group(1)


def load_external_geo(date_str):
    path = f"/data/mrap_bc/{date_str}_mrap.hdr"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    text = read_file(path)
    records = split_records(text)

    geo = {}
    for k, v in records:
        if k in REMOVE_KEYS:
            geo[k] = v
    return geo


def extract_band_names(records):
    for k, v in records:
        if k == "band names":
            return v
    return None


def parse_band_list(band_block):
    if not band_block:
        return []

    # extract inside {...}
    m = re.search(r"\{(.*)\}", band_block, re.S)
    if not m:
        return []

    content = m.group(1).strip()

    # split by commas but keep structure
    parts = [p.strip().rstrip(",") for p in content.split(",") if p.strip()]
    return parts


def rebuild_band_block(bands):
    cleaned = []
    for b in bands:
        b = b.strip().rstrip(",")
        cleaned.append(b)

    return "band names = {" + ",\n".join(cleaned) + "}"


def prefix_bands(bands):
    n = len(bands)

    # rule priority: prefer 4-band grouping, else 3-band grouping
    if n % 4 == 0:
        group = 4
    elif n % 3 == 0:
        group = 3
    else:
        return bands

    out = []
    for i, b in enumerate(bands):
        prefix = ""

        if i < group:
            prefix = "pre "
        elif i < 2 * group:
            prefix = "pst "

        if not b.startswith("pre ") and not b.startswith("pst "):
            b = prefix + b

        out.append(b)

    return out


def clean_records(records):
    cleaned = []
    for k, v in records:
        if k in REMOVE_KEYS:
            continue
        if k == "band names":
            continue
        cleaned.append((k, v))
    return cleaned


def format_records(records):
    lines = []

    # keep ENVI first if present
    lines.append("ENVI")

    for k, v in records:
        if k == "envi":
            continue
        lines.append(f"{k} = {v}")

    return "\n".join(lines)


def main():
    path = sys.argv[1]

    backup_once(path)

    text = read_file(path)
    records = split_records(text)

    # extract band names
    band_block = extract_band_names(records)
    bands = parse_band_list(band_block)

    if not bands:
        raise RuntimeError("No band names found or failed to parse")

    # fix bands
    bands = prefix_bands(bands)
    band_block_fixed = rebuild_band_block(bands)

    # remove broken geo fields
    records = clean_records(records)

    # reinsert corrected band names
    records.append(("band names", band_block_fixed))

    # load external geo info
    date_str = parse_date_from_filename(path)
    geo = load_external_geo(date_str)

    for k in REMOVE_KEYS:
        if k in geo:
            records.append((k, geo[k]))

    # rebuild file
    out = format_records(records)

    # remove empty lines + trailing whitespace
    out = "\n".join(line.rstrip() for line in out.splitlines() if line.strip())

    write_file(path, out)


if __name__ == "__main__":
    main()

