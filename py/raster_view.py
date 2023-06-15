import sys
import numpy as np
from osgeo import gdal
from OpenGL.GL import *
from OpenGL.GLUT import *

# Global variables
image_data = None
image_width = None
image_height = None


def load_image(image_path):
    global image_data, image_width, image_height
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image_width = dataset.RasterXSize
    image_height = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    image_data = band.ReadAsArray(0, 0, image_width, image_height).astype(np.float32)
    dataset = None

def draw_scene():
    glClear(GL_COLOR_BUFFER_BIT)
    normalized_image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    flipped_image_data = np.flipud(normalized_image_data)  # flip the image vertically
    glRasterPos2i(-1, -1)  # draw the image
    glDrawPixels(image_width, image_height, GL_RED, GL_FLOAT, flipped_image_data)
    glutSwapBuffers()

def keyboard_callback(key, x, y):
    if key == b'\x1b':  # Escape key
        glutLeaveMainLoop()

def main():
    if len(sys.argv) < 2:
        print("Please provide the input image file as a command-line argument.")
        sys.exit(1)

    image_path = sys.argv[1]
    load_image(image_path)

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
