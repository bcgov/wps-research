'''20250112 first iteration of nn version of classification approach'''

#!/usr/bin/env python3
import sys, os, warnings, glob, pickle
import numpy as np
from osgeo import gdal, ogr
from joblib import Parallel, delayed
import multiprocessing

# ---------------- config ----------------
MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
WINDOW_STEP = 5
TRAINING_FILE = "training_patches.pkl"

gdal.UseExceptions()
warnings.filterwarnings("ignore")
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_DEBUG', 'OFF')

# ---------- utilities ----------
def print_progress_bar(i, total, prefix='', suffix='', length=40, fill='█'):
    pct = f"{100 * i / total:.1f}"
    filled = int(length * i // total)
    bar = fill * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {pct}% {suffix}', end='\r')
    if i == total:
        print()

# ---------- patch classification ----------
def classify_pixel_nn(padded, y, x, patch, train):
    v = padded[y:y+patch, x:x+patch, :].reshape(-1)

    best_lbl = 0
    best_dist = np.inf

    for lbl in (0, 1):
        T = train[lbl]
        if T.size == 0:
            continue
        D = np.sum((T - v) ** 2, axis=1)
        dmin = D.min()
        if dmin < best_dist:
            best_dist = dmin
            best_lbl = lbl

    return y, x, best_lbl

def classify_by_patch_nn(image, train, patch_size=PATCH_SIZE):
    h, w, _ = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros((h, w), dtype=np.uint8)

    coords = [(y, x) for y in range(h) for x in range(w)]
    total = len(coords)

    n_jobs = multiprocessing.cpu_count()
    step = n_jobs
    completed = 0

    def task(y, x):
        return classify_pixel_nn(padded, y, x, patch_size, train)

    print_progress_bar(0, total, "Classifying (pixels):", "Starting")

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=step)(
        delayed(task)(y, x) for (y, x) in coords
    )

    for i, (y, x, lbl) in enumerate(results, 1):
        out[y, x] = lbl
        if i % step == 0 or i == total:
            print_progress_bar(i, total, "Classifying (pixels):", "Running")

    print_progress_bar(total, total, "Classifying (pixels):", "Done")
    return out

# ---------- training ----------
def compute_training_patches(image, rectangles, labels, patch_size=PATCH_SIZE):
    pad = patch_size // 2
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    h, w, _ = image.shape
    S = {0: [], 1: []}

    for (x0,y0,x1,y1), lbl in zip(rectangles, labels):
        if (x1-x0) < MIN_POLY_DIMENSION or (y1-y0) < MIN_POLY_DIMENSION:
            continue

        x0 = max(0,int(x0)); y0 = max(0,int(y0))
        x1 = min(w,int(x1)); y1 = min(h,int(y1))

        for y in range(y0, y1, WINDOW_STEP):
            for x in range(x0, x1, WINDOW_STEP):
                S[lbl].append(
                    padded[y:y+patch_size, x:x+patch_size, :].reshape(-1)
                )

    return S

# ---------- I/O ----------
def load_image_stack(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(ds.RasterCount)],
        axis=-1
    )
    return data, ds

def save_envi(dataset, classification):
    drv = gdal.GetDriverByName("ENVI")
    h, w = classification.shape
    base,_ = os.path.splitext(dataset.GetDescription())
    out = drv.Create(f"{base}_classification.bin", w, h, 1, gdal.GDT_Float32)
    out.SetGeoTransform(dataset.GetGeoTransform())
    out.SetProjection(dataset.GetProjection())
    out.GetRasterBand(1).WriteArray(classification.astype(np.float32))
    out.FlushCache()
    print(f"[DONE] Saved {base}_classification.bin")

# ---------- shapefile ----------
def parse_coords_img_to_rect(s):
    if not s or str(s).upper() == "NO_RASTER":
        return None
    pts = [p.split(',') for p in str(s).split(';') if p.strip()]
    xs = [int(round(float(p[0]))) for p in pts]
    ys = [int(round(float(p[1]))) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def read_training_from_shapefile(shp):
    ds = ogr.Open(shp,0)
    lyr = ds.GetLayer(0)
    out = {}
    for f in lyr:
        src = f.GetField("SRC_IMAGE")
        cls = f.GetField("CLASS")
        rect = parse_coords_img_to_rect(f.GetField("COORDS_IMG"))
        if not src or rect is None:
            continue
        lbl = 1 if cls.strip().upper()=="POSITIVE" else 0
        out.setdefault(src, {'rectangles':[], 'labels':[]})
        out[src]['rectangles'].append(rect)
        out[src]['labels'].append(lbl)
    return out

# ---------- main ----------
def main():

    # --- classify all ---
    if len(sys.argv) == 1:
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)

        files = sorted(glob.glob("*.tif"))
        for i, fname in enumerate(files, 1):
            print_progress_bar(i-1, len(files), "Images:", "Processing")
            img, ds = load_image_stack(fname)
            cls = classify_by_patch_nn(img, train)
            save_envi(ds, cls)
        print_progress_bar(len(files), len(files), "Images:", "Done")
        return

    shp = sys.argv[1]
    target = sys.argv[2] if len(sys.argv)>2 else None

    # --- classification only ---
    if target:
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)

        img, ds = load_image_stack(target)
        cls = classify_by_patch_nn(img, train)
        save_envi(ds, cls)
        return

    # --- training ---
    training = read_training_from_shapefile(shp)
    T = {0:[], 1:[]}

    for i,(src,d) in enumerate(training.items(),1):
        if not os.path.isfile(src):
            continue
        img,_ = load_image_stack(src)
        S = compute_training_patches(img, d['rectangles'], d['labels'])
        for lbl in (0,1):
            T[lbl].extend(S[lbl])
        print_progress_bar(i,len(training),"Training:", "Done")

    T[0] = np.asarray(T[0], dtype=np.float32)
    T[1] = np.asarray(T[1], dtype=np.float32)

    with open(TRAINING_FILE,"wb") as f:
        pickle.dump(T,f)

    print(f"Saved {TRAINING_FILE}")

if __name__ == "__main__":
    main()

