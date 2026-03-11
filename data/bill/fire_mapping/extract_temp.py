from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from SAFE import extract_and_resample_L1


root_dir = Path("/data/bill/mrap/Level1")
out_dir = Path("/home/bill/GitHub/wps-research/data/bill/C11659/L1C")
out_dir.mkdir(parents=True, exist_ok=True)
safe_dirs = list(root_dir.glob("*.SAFE"))

def process_safe(safe_dir):

    print(f"Processing {safe_dir.name}")
    extract_and_resample_L1(
        safe_dir=safe_dir,
        band_list=['B12', 'B11', 'B09', 'B08'],
        target_resolution=20,
        out_dir=out_dir
    )
    return safe_dir.name


# Adjust max_workers depending on CPU & disk
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_safe, d) for d in safe_dirs]

    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Finished {result}")
        except Exception as e:
            print("Error:", e)