'''20230629 openGL image display
'''
import sys
import numpy as np
from osgeo import gdal
from OpenGL.GL import *
from OpenGL.GLUT import *
import threading
import tkinter as tk
from tkinter import ttk


# Global variables
image_data = None
image_width = None
image_height = None
num_bands = None
slider_value = 1.0


def load_image(image_path):
    global image_data, image_width, image_height, num_bands
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image_width = dataset.RasterXSize
    image_height = dataset.RasterYSize
    num_bands = dataset.RasterCount
    image_data = np.zeros((image_height, image_width, num_bands), dtype=np.float32)

    for i in range(num_bands):
        band = dataset.GetRasterBand(i + 1)
        image_data[:, :, i] = band.ReadAsArray(0, 0, image_width, image_height).astype(np.float32)

    dataset = None


def draw_scene():
    glClear(GL_COLOR_BUFFER_BIT)
    scaled_image_data = image_data * slider_value
    normalized_image_data = (scaled_image_data - np.min(scaled_image_data)) / (
            np.max(scaled_image_data) - np.min(scaled_image_data)
    )
    flipped_image_data = np.flipud(normalized_image_data)  # flip the image vertically
    glRasterPos2i(-1, -1)  # draw the image
    glDrawPixels(image_width, image_height, GL_RGB, GL_FLOAT, flipped_image_data)
    glutSwapBuffers()


def keyboard_callback(key, x, y):
    if key == b'\x1b':  # Escape key
        glutLeaveMainLoop()


def update_slider_value(value):
    global slider_value
    slider_value = float(value)
    glutPostRedisplay()


def tkinter_worker():
    root = tk.Tk()
    root.title("Slider")
    slider_label = ttk.Label(root, text="Scale Channels")
    slider_label.pack(side=tk.TOP)
    slider = ttk.Scale(
        root,
        from_=0.0,
        to=2.0,
        command=update_slider_value,
        orient=tk.HORIZONTAL,
        length=300
    )
    slider.pack(side=tk.TOP)
    root.mainloop()


def main():
    if len(sys.argv) < 2:
        print("Please provide the input image file as a command-line argument.")
        sys.exit(1)

    image_path = sys.argv[1]
    load_image(image_path)

    # Launch Tkinter worker thread
    tkinter_thread = threading.Thread(target=tkinter_worker)
    tkinter_thread.start()

    # OpenGL setup
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    glutInitWindowSize(image_width, image_height)
    glutCreateWindow(b"OpenGL Image Viewer")
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutDisplayFunc(draw_scene)
    glutKeyboardFunc(keyboard_callback)
    glutMainLoop()


if __name__ == "__main__":
    main()

