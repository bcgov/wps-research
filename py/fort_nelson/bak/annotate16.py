

import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal, osr
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
from joblib import Parallel, delayed

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
            cov += np.eye(cov.shape[0]) * 1e-5  # regularize covariance matrix
            mean_covs[label] = (mean, cov)
        else:
            mean_covs[label] = (None, None)
    return mean_covs

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return diff @ inv_cov @ diff.T

def classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov):
    patch = padded[y:y+patch_size, x:x+patch_size, :]
    patch_flat = patch.flatten()

    scores = {}
    for label in [0, 1]:
        mean, _ = mean_covs[label]
        if mean is None or inv_cov[label] is None:
            scores[label] = np.inf
        else:
            scores[label] = mahalanobis_distance(patch_flat, mean, inv_cov[label])

    return 1 if scores[1] < scores[0] else 0

def classify_by_gaussian_parallel(image, mean_covs, patch_size=7, progress_callback=None):
    h, w, b = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    output = np.zeros((h, w), dtype=np.uint8)

    inv_cov = {}
    for label in [0, 1]:
        mean, cov = mean_covs[label]
        if mean is None or cov is None:
            inv_cov[label] = None
        else:
            try:
                inv_cov[label] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov[label] = np.linalg.pinv(cov)

    def classify_row(y):
        row_result = np.zeros(w, dtype=np.uint8)
        for x in range(w):
            row_result[x] = classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov)
        if progress_callback:
            progress_callback()
        return row_result

    results = Parallel(n_jobs=-1, backend="loky")(delayed(classify_row)(y) for y in range(h))

    for y, row in enumerate(results):
        output[y, :] = row
    return output

def save_classification_envi(classified, src_dataset, outname):
    driver = gdal.GetDriverByName('ENVI')
    if os.path.exists(outname):
        driver.Delete(outname)
    out_ds = driver.Create(outname, src_dataset.RasterXSize, src_dataset.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(src_dataset.GetGeoTransform())
    out_ds.SetProjection(src_dataset.GetProjection())

    band = out_ds.GetRasterBand(1)
    band.WriteArray(classified.astype(np.float32))
    band.SetNoDataValue(-9999)
    band.FlushCache()
    out_ds = None
    print(f"Classification saved to {outname} in ENVI format (32-bit float).")

class ENVIImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("ENVI Image Patch Classifier (Probabilistic)")

        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="Classify", command=self.classify).pack(side=tk.LEFT, padx=5, pady=5)

        self.color_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Green / Red", variable=self.color_var).pack(side=tk.LEFT)

        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to start.")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor='w')
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

        self.tk_image = None
        self.rectangles = []  # ((x0, y0, x1, y1), label)
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.dataset = None
        self.full_image_data = None
        self.model = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_image(self):
        filename = filedialog.askopenfilename(title="Open ENVI Image", filetypes=[("ENVI files", "*.bin"), ("All files", "*.*")])
        if not filename:
            return
        self.load_image(filename)

    def load_image(self, path):
        self.dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not self.dataset:
            messagebox.showerror("Error", f"Failed to load ENVI image: {path}")
            self.status_var.set("Failed to load image.")
            return

        band_count = self.dataset.RasterCount
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize

        self.full_image_data = np.stack([
            self.dataset.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
            for i in range(band_count)
        ], axis=-1)

        if band_count >= 3:
            display_data = self.full_image_data[:, :, :3]
        else:
            display_data = np.stack([self.full_image_data[:, :, 0]] * 3, axis=-1)

        rgb = normalize_float32_to_uint8(display_data)
        img = Image.fromarray(rgb, mode='RGB')
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.rectangles.clear()
        self.model = None
        self.status_var.set(f"Loaded image: {os.path.basename(path)} ({width}x{height})")

    def load_model(self):
        pkl_path = filedialog.askopenfilename(title="Load Model (pickle)", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not pkl_path:
            return
        try:
            with open(pkl_path, "rb") as f:
                self.model = pickle.load(f)
            self.status_var.set(f"Loaded model from {os.path.basename(pkl_path)}")
            messagebox.showinfo("Model loaded", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Failed to load model.")

    def on_mouse_down(self, event):
        if self.full_image_data is None:
            return
        self.start_x = int(self.canvas.canvasx(event.x))
        self.start_y = int(self.canvas.canvasy(event.y))
        color = "green" if self.color_var.get() else "red"
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=color, width=2
        )

    def on_mouse_drag(self, event):
        if self.current_rect:
            cur_x = int(self.canvas.canvasx(event.x))
            cur_y = int(self.canvas.canvasy(event.y))
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_up(self, event):
        if self.current_rect:
            coords = list(map(int, self.canvas.coords(self.current_rect)))
            x0, y0, x1, y1 = coords
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])

            label = 1 if self.color_var.get() else 0
            self.rectangles.append(((x0, y0, x1, y1), label))
            self.canvas.itemconfig(self.current_rect, outline="green" if label == 1 else "red")
            self.current_rect = None
            self.status_var.set(f"Added {'green' if label == 1 else 'red'} rectangle at ({x0},{y0})-({x1},{y1}).")

    def classify(self):
        if self.full_image_data is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        if not self.rectangles and self.model is None:
            messagebox.showwarning("No Data", "Draw rectangles or load a model first.")
            return

        try:
            if self.model is None:
                rectangles_coords = [r[0] for r in self.rectangles]
                labels = [r[1] for r in self.rectangles]
                self.status_var.set("Computing patch mean/covariance...")
                print("Computing patch mean/covariance...")
                mean_covs = compute_patch_mean_cov(self.full_image_data, rectangles_coords, labels, patch_size=7)
                print("Patch mean/covariance computed.")
                self.status_var.set("Saving model...")
                model_filename = "patch_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(mean_covs, f)
                print(f"Model saved to {model_filename}")
                self.status_var.set(f"Model saved to {model_filename}")
                self.model = mean_covs
            else:
                mean_covs = self.model

            height, width, _ = self.full_image_data.shape
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

            progress_count = [0]
            total = height

            def progress_update():
                progress_count[0] += 1
                if progress_count[0] % 50 == 0 or progress_count[0] == total:
                    print(f"Classification progress: {progress_count[0]}/{total} rows")

            self.status_var.set("Starting classification...")
            print("Starting classification...")
            classified = classify_by_gaussian_parallel(
                self.full_image_data, mean_covs, patch_size=7,
                progress_callback=progress_update
            )
            print("Classification done.")
            self.status_var.set("Classification done.")

            if width > screen_w or height > screen_h:
                base = os.path.splitext(os.path.basename(self.dataset.GetDescription()))[0]
                outname = f"{base}_classification.bin"
                self.status_var.set(f"Saving classification to {outname} ...")
                print(f"Saving classification to {outname} ...")
                save_classification_envi(classified, self.dataset, outname)
                messagebox.showinfo("Classification saved", f"Classification saved to ENVI file:\n{outname}")
                # Clear canvas image/classification display since image too large
                self.canvas.delete("classification_overlay")
            else:
                # Display classification on canvas in black and white
                bw_img = Image.fromarray(classified * 255).convert('L')  # 0 or 255 pixels
                self.classification_tkimg = ImageTk.PhotoImage(bw_img)
                # Remove old classification overlay if exists
                self.canvas.delete("classification_overlay")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.classification_tkimg, tags="classification_overlay")
                self.status_var.set("Classification displayed on image.")

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_var.set("Classification failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ENVIImageAnnotator(root)
    root.mainloop()

