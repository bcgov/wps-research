

import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal
import numpy as np
from PIL import Image, ImageTk
from joblib import Parallel, delayed
import pickle
import os
import time

# --------------------------- Utility Functions ---------------------------

def normalize_float32_to_uint8(img):
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)
        scaled = (band - min_val) / (max_val - min_val + 1e-6)
        scaled = np.clip(scaled, 0, 1)
        out[:, :, i] = (scaled * 255).astype(np.uint8)
    return out

def compute_patch_mean_cov(image, rectangles, labels, patch_size=7):
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, b = image.shape
    samples = {0: [], 1: []}

    for (x0, y0, x1, y1), label in zip(rectangles, labels):
        x0 = max(0, x0)
        x1 = min(w, x1)
        y0 = max(0, y0)
        y1 = min(h, y1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                patch = padded[y:y+patch_size, x:x+patch_size, :]
                samples[label].append(patch.flatten())

    mean_covs = {}
    for label in [0, 1]:
        if samples[label]:
            data = np.vstack(samples[label])
            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-5  # regularization
            mean_covs[label] = (mean, cov)
        else:
            mean_covs[label] = (None, None)
    return mean_covs

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return diff @ inv_cov @ diff.T

def classify_row(y, image, mean_covs, inv_covs, patch_size):
    h, w, _ = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    row_result = np.zeros(w, dtype=np.uint8)

    for x in range(w):
        patch = padded[y:y+patch_size, x:x+patch_size, :]
        flat = patch.flatten()

        scores = {}
        for label in [0, 1]:
            mean, _ = mean_covs[label]
            inv_cov = inv_covs[label]
            if mean is None or inv_cov is None:
                scores[label] = np.inf
            else:
                scores[label] = mahalanobis_distance(flat, mean, inv_cov)

        row_result[x] = 1 if scores[1] < scores[0] else 0
    return row_result

def classify_image(image, mean_covs, patch_size=7):
    print("Preparing classification...")
    h, w, _ = image.shape
    start = time.time()

    # Precompute inverse covariance matrices
    inv_covs = {}
    for label in [0, 1]:
        mean, cov = mean_covs[label]
        if mean is not None and cov is not None:
            try:
                inv_covs[label] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_covs[label] = np.linalg.pinv(cov)
        else:
            inv_covs[label] = None

    print("Starting classification...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(classify_row)(y, image, mean_covs, inv_covs, patch_size)
        for y in range(h)
    )

    classified = np.vstack(results)
    print(f"Classification completed in {time.time() - start:.1f}s")
    return classified

def save_envi_float32(output_path, classified, reference_dataset):
    driver = gdal.GetDriverByName("ENVI")
    h, w = classified.shape
    out_ds = driver.Create(output_path, w, h, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_ds.SetProjection(reference_dataset.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(classified.astype(np.float32))
    out_ds.FlushCache()
    print(f"Saved classification to {output_path}")

# ---------------------------- GUI Class ----------------------------

class ENVIImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("ENVI Image Annotator + Headless Classifier")

        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control = tk.Frame(root)
        control.pack(fill=tk.X)

        tk.Button(control, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(control, text="Load Model", command=self.load_model).pack(side=tk.LEFT)

        self.color_var = tk.BooleanVar()
        tk.Checkbutton(control, text="Green / Red", variable=self.color_var).pack(side=tk.LEFT)

        self.tk_image = None
        self.rectangles = []
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.dataset = None
        self.image_data = None
        self.filename = None
        self.loaded_model = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("ENVI .bin", "*.bin")])
        if not path:
            return

        self.filename = path
        hdr_path = os.path.splitext(path)[0] + ".hdr"
        self.dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not self.dataset:
            messagebox.showerror("Error", "Failed to open image.")
            return

        self.image_data = np.stack([
            self.dataset.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
            for i in range(self.dataset.RasterCount)
        ], axis=-1)

        h, w = self.image_data.shape[:2]
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        if h > screen_h or w > screen_w:
            print("Large image detected. Classification will be saved, not displayed.")
            self.tk_image = None
            self.canvas.delete("all")
        else:
            display_data = self.image_data[:, :, :3] if self.image_data.shape[2] >= 3 else np.stack([self.image_data[:, :, 0]]*3, axis=-1)
            rgb = normalize_float32_to_uint8(display_data)
            img = Image.fromarray(rgb, mode='RGB')
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.config(width=w, height=h)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.rectangles.clear()

    def on_mouse_down(self, event):
        if not self.tk_image: return
        self.start_x = int(self.canvas.canvasx(event.x))
        self.start_y = int(self.canvas.canvasy(event.y))
        color = "green" if self.color_var.get() else "red"
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=color, width=2)

    def on_mouse_drag(self, event):
        if self.current_rect:
            x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, x, y)

    def on_mouse_up(self, event):
        if self.current_rect:
            x0, y0, x1, y1 = map(int, self.canvas.coords(self.current_rect))
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            label = 1 if self.color_var.get() else 0
            self.rectangles.append(((x0, y0, x1, y1), label))
            self.canvas.itemconfig(self.current_rect, outline="green" if label else "red")
            self.current_rect = None
            self.run_classification()

    def run_classification(self):
        if self.image_data is None or not self.rectangles:
            return

        mean_covs = compute_patch_mean_cov(self.image_data, [r[0] for r in self.rectangles], [r[1] for r in self.rectangles])
        classified = classify_image(self.image_data, mean_covs)

        h, w = classified.shape
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        if h > screen_h or w > screen_w:
            print("Saving classification to ENVI file (float32)...")
            out_path = os.path.splitext(self.filename)[0] + "_classification.bin"
            save_envi_float32(out_path, classified, self.dataset)
        else:
            img = Image.fromarray((classified * 255).astype(np.uint8), mode='L')
            win = tk.Toplevel(self.root)
            win.title("Classification Result")
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(win, image=tk_img)
            label.image = tk_img
            label.pack()

    def load_model(self):
        pkl_path = filedialog.askopenfilename(filetypes=[("Pickle model", "*.pkl")])
        if not pkl_path:
            return

        with open(pkl_path, 'rb') as f:
            self.loaded_model = pickle.load(f)

        if self.image_data is None:
            messagebox.showinfo("Info", "Model loaded. Now load an image.")
            return

        print("Running classification from model...")
        classified = classify_image(self.image_data, self.loaded_model)

        out_path = os.path.splitext(self.filename)[0] + "_classification.bin"
        save_envi_float32(out_path, classified, self.dataset)

# ------------------------- Entry Point -------------------------

if __name__ == "__main__":
    gdal.UseExceptions()
    root = tk.Tk()
    app = ENVIImageAnnotator(root)
    root.mainloop()

