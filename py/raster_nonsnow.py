'''20251021 create stack filtered for summer dates
'''

import re
import numpy as np

# === CONFIG ===
hdr_file = "stack.hdr"
bin_file = "stack.bin"
out_hdr = "stack_nonsnow.hdr"
out_bin = "stack_nonsnow.bin"
summer_months = {"05", "06", "07", "08", "09"}  # May–September only

# === READ HEADER ===
with open(hdr_file, "r") as f:
    hdr_text = f.read()

# Extract key metadata
samples = int(re.search(r"samples\s*=\s*(\d+)", hdr_text).group(1))
lines = int(re.search(r"lines\s*=\s*(\d+)", hdr_text).group(1))
bands = int(re.search(r"bands\s*=\s*(\d+)", hdr_text).group(1))
interleave = re.search(r"interleave\s*=\s*(\w+)", hdr_text).group(1).lower()
datatype = int(re.search(r"data type\s*=\s*(\d+)", hdr_text).group(1))

# Sanity check
if interleave != "bsq":
    raise ValueError("This script currently supports only BSQ interleaved data.")

# Extract band names
band_match = re.search(r"band names\s*=\s*\{(.*?)\}", hdr_text, re.S)
if not band_match:
    raise ValueError("No 'band names' section found in header.")
band_names = [b.strip() for b in band_match.group(1).split(",") if b.strip()]

if len(band_names) != bands:
    print("Warning: Number of band names does not match 'bands' count in header.")

# === FILTER SUMMER MONTHS ===
selected_indices = []
selected_bandnames = []
for i, b in enumerate(band_names):
    m = re.match(r"(\d{8})/", b)
    if not m:
        continue
    date = m.group(1)
    month = date[4:6]
    if month in summer_months:
        selected_indices.append(i)
        selected_bandnames.append(b)

print(f"Selected {len(selected_indices)} of {bands} bands (May–September).")

# === READ AND WRITE BANDS ===
band_size = samples * lines  # number of pixels per band
dtype = np.float32  # ENVI data type 4 = float32

with open(bin_file, "rb") as fin, open(out_bin, "wb") as fout:
    for new_i, old_i in enumerate(selected_indices):
        offset = old_i * band_size * 4  # 4 bytes per float
        fin.seek(offset)
        band_data = np.fromfile(fin, dtype=dtype, count=band_size)

        # Write to new file
        band_data.tofile(fout)

        # Print progress
        current_bandname = selected_bandnames[new_i]
        print(f"Writing band {new_i+1}/{len(selected_indices)}: {current_bandname}")

# === CREATE NEW HEADER ===
# Replace band count and band names in the header text
new_hdr_text = re.sub(r"bands\s*=\s*\d+", f"bands = {len(selected_indices)}", hdr_text)
new_band_list = ",\n".join(selected_bandnames)
new_hdr_text = re.sub(
    r"band names\s*=\s*\{(.*?)\}",
    f"band names = {{{new_band_list}}}",
    new_hdr_text,
    flags=re.S,
)

# Write new header file
with open(out_hdr, "w") as f:
    f.write(new_hdr_text)

print("\n✅ Done.")
print(f"Created: {out_bin} and {out_hdr}")



