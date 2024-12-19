'''20241218 sample of embedding multispectral image data (SWIR example) into 2d 
usage: 

python3 embedding_test.py small/stack.bin umap
# python3 embedding_test.py small/G80223_20230513.bin_scale.bin

'''

import os
import sys
import umap
import math
import pickle
import rasterio
import warnings
import numpy as np
from sklearn.manifold import TSNE  # Import t-SNE from scikit-learn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backend_bases import MouseEvent
from matplotlib.path import Path
warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform")

# Function to choose between UMAP and t-SNE
def get_model(model_type=None):
    if model_type == 'umap':
        return umap.UMAP(n_components=2)
    elif model_type == 'tsne':
        return TSNE(n_components=2, perplexity=22)
    else:
        raise ValueError("model_type should be either 'umap' or 'tsne'")

# Load raster data
raster_file = sys.argv[1]
data, width, height, num_bands, transform, crs = None, None, None, None, None, None

with rasterio.open(raster_file) as src:
    data = src.read()  # This reads the raster into a 3D array (bands x height x width)
    width = src.width  # Number of columns (pixels in x)
    height = src.height  # Number of rows (pixels in y)
    num_bands = src.count  # Number of bands (e.g., 3 for RGB or more for multispectral)
    transform = src.transform
    crs = src.crs

# Reshape data for UMAP or t-SNE (flatten the image)
reshaped_data = data.reshape(num_bands, -1).T  # Shape: (num_pixels, num_bands)
reshaped_data = np.nan_to_num(reshaped_data)

# Choose model type (UMAP or t-SNE)
model_type = sys.argv[2] if len(sys.argv) > 2 else 'tsne'  # Default to 'umap' if not specified
pkl_exist = os.path.exists('model.pkl')

model = get_model(model_type)

# Apply chosen model for dimensionality reduction (project to 2D)
print("embedding..", model_type)
embedding = model.fit_transform(reshaped_data) if not pkl_exist else pickle.load(open('model.pkl', 'rb'))
print("embedding loaded.")

if not pkl_exist:
    pickle.dump(embedding, open('model.pkl', 'wb'))

# If the raster has 3 bands (RGB), we use the RGB values to color the projection
if True: 
    # Normalize the data for proper RGB scaling (values between 0 and 1)
    rgb_values = np.moveaxis(data[:3, :, :], 0, -1).reshape(-1, 3)  # Reshape (height * width, 3) for RGB
    print(rgb_values.shape)

    for i in range(3):
        values = rgb_values[:, i].ravel().tolist()
        values.sort()
        n_pct = 1.5  # percent for stretch value
        frac = n_pct / 100.
        rgb_min, rgb_max = values[int(math.floor(float(len(values)) * frac))], \
                           values[int(math.floor(float(len(values)) * (1. - frac)))]

        # Normalize RGB channels
        rgb_values[:, i] -= rgb_min
        rgb_values[:, i] /= (rgb_max - rgb_min)
        rgb_values[:, i][np.where(rgb_values[:, i] < 0.)] = 0.
        rgb_values[:, i][np.where(rgb_values[:, i] > 1.)] = 1.

# Create a figure with three subplots (for UMAP/t-SNE, RGB values with masks, and original image)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# First subplot: UMAP or t-SNE projection colored by RGB values
scatter = axs[0].scatter(embedding[:, 0], embedding[:, 1], c=rgb_values, s=1, alpha=0.5)  # Scatter plot with RGB coloring
axs[0].set_title(f'{model_type.upper()} Projection of Raster Data')
axs[0].set_xlabel(f'{model_type.upper()} Component 1')
axs[0].set_ylabel(f'{model_type.upper()} Component 2')

# Second subplot: Display RGB values using imshow with the mask applied
# Reshape the RGB values back to (height, width, 3) for imshow
rgb_image = rgb_values.reshape(height, width, 3)  # Recreate the RGB image

# Show the RGB image using imshow
axs[1].imshow(rgb_image)
axs[1].axis('off')  # Hide axes for better visualization
axs[1].set_title('RGB Values of Raster')

# Third subplot: Original image (without any modifications)
# Display the original image (in this case, we are using RGB bands for visualization)
original_image = np.moveaxis(data[:3, :, :], 0, -1)  # First 3 bands for the original image

# Display the original image
axs[2].imshow(original_image)
axs[2].axis('off')  # Hide axes for better visualization
axs[2].set_title('Original Image')

# Initialize variables for polygon drawing
polygon_points = []
polygon_path = None
drawing_polygon = False

def on_click(event):
    global polygon_points, polygon_path, drawing_polygon

    # Check if we are within the first (left) pane
    if event.inaxes != axs[0]:
        return

    if event.button == 1:  # Left click adds a new vertex to the polygon
        if drawing_polygon:
            polygon_points.append([event.xdata, event.ydata])
            # Plot the new red line segment
            axs[0].plot([polygon_points[-2][0], polygon_points[-1][0]],
                        [polygon_points[-2][1], polygon_points[-1][1]], 'r-', alpha=0.7)
            axs[0].figure.canvas.draw()
        else:
            # Start drawing the polygon
            polygon_points = [[event.xdata, event.ydata]]
            drawing_polygon = True
    elif event.button == 3:  # Right click to close the polygon
        if drawing_polygon and len(polygon_points) > 2:
            polygon_points.append(polygon_points[0])  # Close the polygon
            # Draw the final segment to close the polygon
            axs[0].plot([polygon_points[-2][0], polygon_points[-1][0]],
                        [polygon_points[-2][1], polygon_points[-1][1]], 'r-', alpha=0.7)
            axs[0].figure.canvas.draw()
            drawing_polygon = False
            # Create the path from the polygon vertices
            polygon_path = Path(polygon_points)
            mask_polygon()

def mask_polygon():
    global polygon_path, rgb_image, embedding, scatter
    # Create a mask from the polygon path
    inside_polygon = polygon_path.contains_points(embedding)

    # Plot the points inside the polygon on the left (embedding space)
    axs[0].scatter(embedding[inside_polygon, 0], embedding[inside_polygon, 1],
                   color='green', marker='x', label="Points Inside Polygon", alpha=0.7)
    axs[0].legend()

    # Create a mask array with the same number of pixels as the original image
    inside_polygon_image = np.zeros((height, width), dtype=bool)

    # Use the indices from the embedding to map them back to the original image grid
    for i, is_inside in enumerate(inside_polygon):
        if is_inside:
            row, col = divmod(i, width)  # Map the flat index to row, col in the image grid
            inside_polygon_image[row, col] = True

    # Create the masked image
    masked_image = np.copy(rgb_image)
    masked_image[~inside_polygon_image] = [0, 0, 0]  # Masked area is black

    # Optionally, change the masked color (e.g., red)
    masked_image[inside_polygon_image] = [255, 0, 0]  # Red mask

    # Display the masked image on the right
    axs[1].imshow(masked_image)
    axs[1].set_title('Masked Image')
    axs[1].axis('off')
    axs[1].figure.canvas.draw()

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show()


