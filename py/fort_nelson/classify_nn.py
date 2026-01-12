'''20250112 first iteration of nn version of classification approach'''

#!/usr/bin/env python3
import sys, os, warnings, glob, pickle
import numpy as np
from osgeo import gdal, ogr
from joblib import Parallel, delayed
import multiprocessing as mp

# ---------------- config ----------------
MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
WINDOW_STEP = 5
TRAINING_FILE = "training_patches.pkl"

single_thread = False

gdal.UseExceptions()
warnings.filterwarnings("ignore")
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_DEBUG', 'OFF')

# ---------- progress ----------
def print_progress_bar(i, total, prefix='', suffix='', length=40):
    pct = 100.0 * i / total
    filled = int(length * i // total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {pct:5.1f}% {suffix}', end='')
    if i >= total:
        print()

# ---------- parfor abstraction ----------
def parfor(my_function, my_inputs, n_thread=min(32, int(mp.cpu_count()))):
    print(f"PARFOR using {n_thread} threads")
    if n_thread == 1 or single_thread:
        return [my_function(x) for x in my_inputs]

    if my_inputs is None or len(my_inputs) == 0:
        return []

    return Parallel(n_jobs=n_thread, backend="loky")(
        delayed(my_function)(x) for x in my_inputs
    )

# ---------- pixel classification ----------
def classify_pixel_nn(args):
    padded, y, x, patch, train = args

    v = padded[y:y+patch, x:x+patch, :].reshape(-1)

    best_lbl = 0
    best_dist = np.inf

    for lbl in (0, 1):
        T = train[lbl]
        if T.size == 0:
            continue
        d = np.sum((T - v) ** 2, axis=1).min()
        if d < best_dist:
            best_dist = d
            best_lbl = lbl

    return y, x, best_lbl

def classify_by_patch_nn(image, train, patch_size=PATCH_SIZE):
    h, w, _ = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

    out = np.zeros((h, w), dtype=np.uint8)

    coords = [(padded, y, x, patch_size, train)
              for y in range(h) for x in range(w)]

    total = len(coords)
    n_thread = min(32, mp.cpu_count())

    print_progress_bar(0, total, "Classifying (pixels):", "starting")

    results = []
    chunk = n_thread
    for i in range(0, total, chunk):
        batch = coords[i:i+chunk]
        r = parfor(classify_pixel_nn, batch, n_thread)
        results.extend(r)
        print_progress_bar(min(i+chunk, total), total,
                           "Classifying (pixels):", "running")

    for y, x, lbl in results:
        out[y, x] = lbl

    print_progress_bar(total, total, "Classifying (pixels):", "done")
    return out

# ---------- training ----------
def compute_training_patches(image, rectangles, labels):
    pad = PATCH_SIZE // 2
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
                    padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :].reshape(-1)
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
    print(f"[DONE] {base}_classification.bin")

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

    if len(sys.argv) == 1:
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)

        files = sorted(glob.glob("*.tif"))
        for i,fname in enumerate(files,1):
            print_progress_bar(i-1,len(files),"Images:","processing")
            img, ds = load_image_stack(fname)
            cls = classify_by_patch_nn(img, train)
            save_envi(ds, cls)
        print_progress_bar(len(files),len(files),"Images:","done")
        return

    shp = sys.argv[1]
    target = sys.argv[2] if len(sys.argv)>2 else None

    if target:
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)
        img, ds = load_image_stack(target)
        cls = classify_by_patch_nn(img, train)
        save_envi(ds, cls)
        return

    training = read_training_from_shapefile(shp)
    T = {0:[], 1:[]}

    for i,(src,d) in enumerate(training.items(),1):
        if not os.path.isfile(src):
            continue
        img,_ = load_image_stack(src)
        S = compute_training_patches(img, d['rectangles'], d['labels'])
        for lbl in (0,1):
            T[lbl].extend(S[lbl])
        print_progress_bar(i,len(training),"Training:","running")

    T[0] = np.asarray(T[0], np.float32)
    T[1] = np.asarray(T[1], np.float32)

    with open(TRAINING_FILE,"wb") as f:
        pickle.dump(T,f)

    print(f"Saved {TRAINING_FILE}")

if __name__ == "__main__":
    main()

