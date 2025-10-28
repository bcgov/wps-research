

import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
from joblib import Parallel, delayed
import time

class ENVIImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("ENVI Image Patch Classifier (Probabilistic)")

        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT)

        self.color_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Green / Red", variable=self.color_var).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tk_image = None
        self.rectangles = []
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        self.dataset = None
        self.full_image_data = None
        self.model_data = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_image(self):
        path = filedialog.askopenfilename(title="Select ENVI .bin file", filetypes=[("ENVI .bin files", "*.bin")])
        if not path:
            return
        self.load_image(path)

    def load_image(self, path):
        self.dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not self.dataset:
            messagebox.showerror("Error", "Failed to load ENVI image.")
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
        self.root.title(f"ENVI Image Patch Classifier - {os.path.basename(path)}")

    def load_model(self):
        pkl_path = filedialog.askopenfilename(title="Select Model Pickle File", filetypes=[("Pickle Files", "*.pkl")])
        if not pkl_path:
            return
        with open(pkl_path, "rb") as f:
            self.model_data = pickle.load(f)
        messagebox.showinfo("Model Loaded", "Patch model loaded successfully.")

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
            coords = list(map(int, self.canvas.coords(self.current_rect)))
            x0, y0, x1, y1 = coords
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])

            label = 1 if self.color_var.get() else 0
            self.rectangles.append(((x0, y0, x1, y1), label))
            self.canvas.itemconfig(self.current_rect, outline="green" if label == 1 else "red")
            self.current_rect = None

            self.show_classification_result()

    def show_classification_result(self):
        if self.full_image_data is None or not self.rectangles:
            return

        labels = [r[1] for r in self.rectangles]
        if 0 not in labels or 1 not in labels:
            print("Need at least one red and one green rectangle to classify.")
            return

        width, height = self.dataset.RasterXSize, self.dataset.RasterYSize
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        coords = [r[0] for r in self.rectangles]
        mean_covs = self.compute_patch_mean_cov(self.full_image_data, coords, labels, patch_size=7)
        classified = self.classify_by_gaussian_parallel(self.full_image_data, mean_covs, patch_size=7)

        if width > screen_width or height > screen_height:
            filename = os.path.splitext(self.dataset.GetDescription())[0] + "_classification.bin"
            self.save_classification_envi(classified, filename)
            print("Classification written to:", filename)
        else:
            img = (classified * 255).astype(np.uint8)
            pil_image = Image.fromarray(img, mode='L')
            win = tk.Toplevel(self.root)
            win.title("Classification Result")
            tk_img = ImageTk.PhotoImage(pil_image)
            label = tk.Label(win, image=tk_img)
            label.image = tk_img
            label.pack()

    def compute_patch_mean_cov(self, image, rectangles, labels, patch_size=7):
        pad = patch_size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        samples = {0: [], 1: []}
        for (x0, y0, x1, y1), label in zip(rectangles, labels):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    patch = padded[y:y+patch_size, x:x+patch_size, :]
                    samples[label].append(patch.flatten())

        mean_covs = {}
        for label in [0, 1]:
            if samples[label]:
                data = np.vstack(samples[label])
                mean = np.mean(data, axis=0)
                cov = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-5
                mean_covs[label] = (mean, cov)
            else:
                mean_covs[label] = (None, None)
        return mean_covs

    def classify_patch(self, y, image, mean_covs, inv_covs, patch_size):
        pad = patch_size // 2
        h, w, _ = image.shape
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        row_result = np.zeros(w, dtype=np.uint8)
        for x in range(w):
            patch = padded[y:y+patch_size, x:x+patch_size, :]
            patch_flat = patch.flatten()
            scores = {label: np.inf for label in [0, 1]}
            for label in [0, 1]:
                mean, _ = mean_covs[label]
                if mean is not None and inv_covs[label] is not None:
                    diff = patch_flat - mean
                    scores[label] = diff @ inv_covs[label] @ diff.T
            row_result[x] = 1 if scores[1] < scores[0] else 0
        return row_result

    def classify_by_gaussian_parallel(self, image, mean_covs, patch_size=7):
        print("Starting classification...")
        start = time.time()

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

        h, w, _ = image.shape
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.classify_patch)(y, image, mean_covs, inv_covs, patch_size)
            for y in range(h)
        )

        output = np.vstack(results)
        print(f"Classification done in {time.time() - start:.2f} seconds.")
        return output

    def save_classification_envi(self, classified, filename):
        driver = gdal.GetDriverByName("ENVI")
        out_ds = driver.Create(filename, classified.shape[1], classified.shape[0], 1, gdal.GDT_Float32)
        if out_ds is None:
            raise RuntimeError("Failed to create output dataset.")

        out_ds.SetGeoTransform(self.dataset.GetGeoTransform())
        out_ds.SetProjection(self.dataset.GetProjection())

        out_band = out_ds.GetRasterBand(1)
        float_data = classified.astype(np.float32)
        out_band.WriteArray(float_data)
        out_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None

        print(f"Classification ENVI file saved to {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ENVIImageAnnotator(root)
    root.mainloop()


