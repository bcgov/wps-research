'''20250112 first iteration of nn version of classification approach'''

#!/usr/bin/env python3
import sys, os, warnings, glob, pickle
import numpy as np
from osgeo import gdal, ogr
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
import time

# ---------------- CONFIG ----------------
MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
WINDOW_STEP = 5
TRAINING_FILE = "training_patches.pkl"

# ---------------------------------------
gdal.UseExceptions()
warnings.filterwarnings("ignore")
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_DEBUG', 'OFF')

# ---------- STATUS ----------
def status(msg):
    print(f"[STATUS] {msg}", flush=True)

def progress(i, total, label):
    pct = 100.0 * i / total
    bar = int(40 * i // total)
    print(f"\r[{label}] |{'█'*bar}{'-'*(40-bar)}| {pct:5.1f}%", end='', flush=True)
    if i == total:
        print()

# ---------- SHARED GLOBALS ----------
SHM_IMAGE = None
SHM_TRAIN = None
IMG_SHAPE = None
TRAIN_SHAPES = None

# ---------- WORKER INIT ----------
def worker_init(img_name, img_shape, train_name, train_shapes):
    global SHM_IMAGE, SHM_TRAIN, IMG_SHAPE, TRAIN_SHAPES
    IMG_SHAPE = img_shape
    TRAIN_SHAPES = train_shapes

    status("Worker attaching to shared memory")
    SHM_IMAGE = shared_memory.SharedMemory(name=img_name)
    SHM_TRAIN = {
        0: shared_memory.SharedMemory(name=train_name[0]),
        1: shared_memory.SharedMemory(name=train_name[1])
    }

# ---------- WORKER TASK ----------
def classify_pixel_task(task):
    y, x = task
    pad = PATCH_SIZE // 2

    image = np.ndarray(IMG_SHAPE, dtype=np.float32, buffer=SHM_IMAGE.buf)
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

    train = {
        0: np.ndarray(TRAIN_SHAPES[0], dtype=np.float32, buffer=SHM_TRAIN[0].buf),
        1: np.ndarray(TRAIN_SHAPES[1], dtype=np.float32, buffer=SHM_TRAIN[1].buf)
    }

    v = padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :].reshape(-1)

    best_lbl = 0
    best_dist = np.inf

    for lbl in (0, 1):
        if train[lbl].size == 0:
            continue
        d = np.sum((train[lbl] - v) ** 2, axis=1).min()
        if d < best_dist:
            best_dist = d
            best_lbl = lbl

    return y, x, best_lbl

# ---------- TRAINING ----------
def compute_training_patches(image, rectangles, labels):
    status("Extracting training patches")
    pad = PATCH_SIZE // 2
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    h, w, _ = image.shape
    S = {0: [], 1: []}

    for (x0,y0,x1,y1), lbl in zip(rectangles, labels):
        if (x1-x0) < MIN_POLY_DIMENSION or (y1-y0) < MIN_POLY_DIMENSION:
            continue

        for y in range(y0, y1, WINDOW_STEP):
            for x in range(x0, x1, WINDOW_STEP):
                S[lbl].append(
                    padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :].reshape(-1)
                )

    return S

# ---------- IO ----------
def load_image_stack(path):
    status(f"Loading image {path}")
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(ds.RasterCount)],
        axis=-1
    )
    return data, ds

def save_envi(dataset, classification):
    status("Writing ENVI classification output")
    drv = gdal.GetDriverByName("ENVI")
    h, w = classification.shape
    base,_ = os.path.splitext(dataset.GetDescription())
    out = drv.Create(f"{base}_classification.bin", w, h, 1, gdal.GDT_Float32)
    out.SetGeoTransform(dataset.GetGeoTransform())
    out.SetProjection(dataset.GetProjection())
    out.GetRasterBand(1).WriteArray(classification.astype(np.float32))
    out.FlushCache()
    status("Output written successfully")

# ---------- SHAPEFILE ----------
def parse_coords_img_to_rect(s):
    if not s or str(s).upper() == "NO_RASTER":
        return None
    pts = [p.split(',') for p in str(s).split(';') if p.strip()]
    xs = [int(round(float(p[0]))) for p in pts]
    ys = [int(round(float(p[1]))) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def read_training_from_shapefile(shp):
    status("Parsing shapefile")
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
    status("Shapefile parsing complete")
    return out

# ---------- CLASSIFICATION ----------
def classify_image_shared(image, train):
    status("Allocating shared memory for image")
    shm_img = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shm_arr = np.ndarray(image.shape, dtype=np.float32, buffer=shm_img.buf)
    shm_arr[:] = image[:]

    shm_train = {}
    for lbl in (0,1):
        shm = shared_memory.SharedMemory(create=True, size=train[lbl].nbytes)
        arr = np.ndarray(train[lbl].shape, dtype=np.float32, buffer=shm.buf)
        arr[:] = train[lbl][:]
        shm_train[lbl] = shm

    h, w, _ = image.shape
    tasks = [(y,x) for y in range(h) for x in range(w)]
    out = np.zeros((h,w), dtype=np.uint8)

    ncpu = mp.cpu_count()
    status(f"Starting worker pool with {ncpu} processes")

    pool = mp.Pool(
        processes=ncpu,
        initializer=worker_init,
        initargs=(shm_img.name, image.shape,
                  {0:shm_train[0].name,1:shm_train[1].name},
                  {0:train[0].shape,1:train[1].shape})
    )

    completed = 0
    total = len(tasks)

    status("Beginning classification")
    for y,x,lbl in pool.imap_unordered(classify_pixel_task, tasks, chunksize=1):
        out[y,x] = lbl
        completed += 1
        if completed % ncpu == 0:
            progress(completed, total, "Classifying")

    pool.close()
    pool.join()

    progress(total, total, "Classifying")
    status("Classification complete")

    shm_img.unlink()
    shm_train[0].unlink()
    shm_train[1].unlink()

    return out

# ---------- MAIN ----------
def main():
    status("Program started")

    if len(sys.argv) == 1:
        status("Mode: classify all images")
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)

        for fname in glob.glob("*.tif"):
            img, ds = load_image_stack(fname)
            cls = classify_image_shared(img, train)
            save_envi(ds, cls)
        return

    shp = sys.argv[1]
    target = sys.argv[2] if len(sys.argv)>2 else None

    if target:
        status("Mode: classify single image")
        print("loading pickle..")
        with open(TRAINING_FILE,"rb") as f:
            train = pickle.load(f)
        print("load image..")
        img, ds = load_image_stack(target)
        print("classify...")
        cls = classify_image_shared(img, train)
        print("save image..")
        save_envi(ds, cls)
        return

    status("Mode: training")
    training = read_training_from_shapefile(shp)
    T = {0:[], 1:[]}

    for i,(src,d) in enumerate(training.items(),1):
        img,_ = load_image_stack(src)
        S = compute_training_patches(img, d['rectangles'], d['labels'])
        for lbl in (0,1):
            T[lbl].extend(S[lbl])
        progress(i,len(training),"Training")

    T[0] = np.asarray(T[0], np.float32)
    T[1] = np.asarray(T[1], np.float32)

    with open(TRAINING_FILE,"wb") as f:
        pickle.dump(T,f)

    status("Training complete")

if __name__ == "__main__":
    main()


