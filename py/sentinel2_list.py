'''20240723 stack sentinel2 .bin files and/or yyyymmdd_mrap.bin files in temporal order
one tile only supported'''
import os
import re

entries = []

# Pattern 1: S2*_yyyymmdd_*.bin  (date in 3rd underscore-delimited field, index 2)
for line in os.popen('ls -1 S2*.bin 2>/dev/null').readlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split('_')
    if len(parts) >= 3 and re.fullmatch(r'\d{8}', parts[2]):
        entries.append((parts[2], line))

# Pattern 2: yyyymmdd_mrap.bin  (date is the first field)
for line in os.popen('ls -1 *_mrap.bin 2>/dev/null').readlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split('_')
    if len(parts) >= 1 and re.fullmatch(r'\d{8}', parts[0]):
        entries.append((parts[0], line))

# Sort by date, then filename for stability
entries.sort()
lines = [e[1] for e in entries]

for line in lines:
    print(line)
print(" ".join(lines))

# Report date range
if entries:
    print(f"\nFirst date: {entries[0][0]}")
    print(f"Last date:  {entries[-1][0]}")
else:
    print("\nNo matching .bin files found.")
