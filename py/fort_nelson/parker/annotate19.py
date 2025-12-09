

import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal
import numpy as np
from PIL import Image, ImageTk
import pickle
from joblib import Parallel, delayed
import sys
import os

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov):
    patch = padded[y:y+patch_size, x:x+patch_size, :]
    patch_flat = patch.flatten()
    scores = {}
    for label in [0, 1]:
        mean, _ = mean_covs[label]
        if mean is None or inv_cov[label] is None:
            scores[label] = np.inf
        else:
            diff = patch_flat - mean
            scores[label] = diff @ inv_cov[label] @ diff.T
    return 1 if scores[1] < scores[0] else 0

def classify_by_gaussian_parallel(image, mean_covs, patch_size=7):
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
        return row_result

    print("Starting classification using all CPU cores...")
    results = Parallel(n_jobs=-1, backend="loky")(delayed(classify_row)(y) for y in range(h))

    for y, row in enumerate(results, 1):
        output[y-1, :] = row
        print_progress_bar(y, h, prefix='Classification Progress:', suffix='Complete', length=50)
    print("Classification completed.")
    return output

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
            cov += np.eye(cov.shape[0]) * 1e-5  # regularize
            mean_covs[label] = (mean, cov)
        else:
            mean_covs[label] = (None, None)
    return mean_covs

def save_envi_classification(filename, classification, dataset):
    # Save classification as 32-bit float ENVI file with same geotransform/projection
    driver = gdal.GetDriverByName('ENVI')
    h, w = classification.shape
    out_file = filename.replace('.bin', '_classification.bin')
    out_ds = driver.Create(out_file, w, h, 1, gdal.GDT_Float32)
    if out_ds is None:
        print(f"Error creating output file {out_file}")
        return
    out_ds.SetGeoTransform(dataset.GetGeoTransform())
    out_ds.SetProjection(dataset.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(classification.astype(np.float32))
    out_band.FlushCache()
    out_ds.FlushCache()
    print(f"Classification saved to {out_file}")

class ENVIImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("ENVI Image Patch Classifier (Probabilistic)")

        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        self.color_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Green / Red", variable=self.color_var).pack(side=tk.LEFT)

        self.toggle_var = tk.BooleanVar(value=False)
        self.toggle_btn = tk.Checkbutton(control_frame, text="Show Classification", variable=self.toggle_var, command=self.update_display)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Classify", command=self.classify).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor='w')
        self.status_label.pack(fill=tk.X)

        self.tk_image = None
        self.class_image = None
        self.rectangles = []  # ((x0, y0, x1, y1), label)
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.dataset = None
        self.full_image_data = None
        self.mean_covs = None
        self.classification_result = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Load default image on start
        self.load_image("sub.bin")

    def load_image(self, path):
        self.dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not self.dataset:
            messagebox.showerror("Error", f"Failed to load ENVI image: {path}")
            self.status_var.set(f"Failed to load image: {path}")
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

        rgb = self.normalize_float32_to_uint8(display_data)
        img = Image.fromarray(rgb, mode='RGB')
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.rectangles.clear()
        self.mean_covs = None
        self.classification_result = None
        self.class_image = None
        self.status_var.set(f"Loaded image {os.path.basename(path)} ({width}x{height})")
        self.toggle_var.set(False)

    def normalize_float32_to_uint8(self, img):
        out = np.zeros_like(img, dtype=np.uint8)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)
            scaled = (band - min_val) / (max_val - min_val + 1e-6)
            scaled = np.clip(scaled, 0, 1)
            out[:, :, i] = (scaled * 255).astype(np.uint8)
        return out

    def on_mouse_down(self, event):
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
            x0, y0, x1, y1 = self.canvas.coords(self.current_rect)
            x0, x1 = int(min(x0, x1)), int(max(x0, x1))
            y0, y1 = int(min(y0, y1)), int(max(y0, y1))
            label = 1 if self.color_var.get() else 0
            self.rectangles.append(((x0, y0, x1, y1), label))
            self.status_var.set(f"Added rectangle {len(self.rectangles)} (Label {label})")
            self.current_rect = None

            # After adding new rectangle, recompute mean_covs
            if self.full_image_data is not None:
                self.mean_covs = compute_patch_mean_cov(self.full_image_data, [r for r, _ in self.rectangles], [l for _, l in self.rectangles])
                self.status_var.set("Patch statistics updated after rectangle added.")
                self.classification_result = None
                self.class_image = None
                self.toggle_var.set(False)
                self.update_display()

    def open_image(self):
        path = filedialog.askopenfilename(title="Open ENVI Image", filetypes=[("ENVI files", "*.bin")])
        if path:
            self.load_image(path)

    def load_model(self):
        path = filedialog.askopenfilename(title="Load Model (PKL)", filetypes=[("Pickle files", "*.pkl")])
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                self.mean_covs = pickle.load(f)
            self.status_var.set(f"Model loaded from {os.path.basename(path)}")
            self.classification_result = None
            self.class_image = None
            self.toggle_var.set(False)
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.status_var.set(f"Failed to load model: {e}")

    def classify(self):
        if self.full_image_data is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return
        if not self.rectangles and not self.mean_covs:
            messagebox.showwarning("Warning", "No training data or model loaded.")
            return

        if not self.mean_covs:
            self.mean_covs = compute_patch_mean_cov(self.full_image_data, [r for r, _ in self.rectangles], [l for _, l in self.rectangles])
            self.status_var.set("Patch statistics computed.")

        # Save model to pkl file
        model_filename = filedialog.asksaveasfilename(title="Save Model As", defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if model_filename:
            try:
                with open(model_filename, 'wb') as f:
                    pickle.dump(self.mean_covs, f)
                self.status_var.set(f"Model saved to {os.path.basename(model_filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")
                return

        h, w, _ = self.full_image_data.shape
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # If image too big, don't show, just save classification
        if w > screen_w or h > screen_h:
            self.status_var.set("Image too large for display. Running classification and saving to file...")
            self.root.update()

            classification = classify_by_gaussian_parallel(self.full_image_data, self.mean_covs)
            save_envi_classification(self.dataset.GetDescription(), classification, self.dataset)
            self.status_var.set("Classification saved. No display due to image size.")
            self.classification_result = classification
            self.class_image = None
            self.toggle_var.set(False)
            self.update_display()
        else:
            self.status_var.set("Running classification...")
            self.root.update()
            classification = classify_by_gaussian_parallel(self.full_image_data, self.mean_covs)
            self.classification_result = classification
            self.status_var.set("Classification done.")
            self.class_image = self.create_classification_image(classification)
            self.toggle_var.set(True)
            self.update_display()

    def create_classification_image(self, classification):
        # Display classification as black (0) and white (1)
        img = Image.fromarray((classification * 255).astype(np.uint8), mode='L')
        return ImageTk.PhotoImage(img)

    def update_display(self):
        self.canvas.delete("all")
        if self.toggle_var.get() and self.class_image is not None:
            self.canvas.config(width=self.class_image.width(), height=self.class_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.class_image)
        elif self.tk_image is not None:
            self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Draw rectangles on top
        for (x0, y0, x1, y1), label in self.rectangles:
            color = "green" if label == 1 else "red"
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = ENVIImageAnnotator(root)
    root.mainloop()

