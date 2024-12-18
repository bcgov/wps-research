'''20241218 sample of embedding multispectral image data (SWIR example) into 2d 
usage: 

python3 embedding_test.py small/G80223_20230513.bin_scale.bin

'''
import os
import sys
import umap
import math
import pickle
import rasterio
import warnings 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Import t-SNE from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform")

# Function to choose between UMAP and t-SNE
def get_model(model_type='tsne'):
    print("fitting model..")
    if model_type == 'umap':
        return umap.UMAP(n_components=2)
    elif model_type == 'tsne':
        return TSNE(n_components=2, perplexity = 111) 
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

# Choose model type (UMAP or t-SNE)
model_type = sys.argv[2] if len(sys.argv) > 2 else 'tsne'  # Default to 'umap' if not specified
pkl_exist = os.path.exists('model.pkl')

if pkl_exist:
    print("reloading model from pkl..")
model = pickle.load(open('model.pkl', 'rb')) if pkl_exist else get_model(model_type)

if not pkl_exist:
    pickle.dump(model, open('model.pkl', 'wb'))

# Apply chosen model for dimensionality reduction (project to 2D)
embedding = model.fit_transform(reshaped_data)

# If the raster has 3 bands (RGB), we use the RGB values to color the projection
if num_bands == 3:
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

else:
    # If it's not RGB, use the first component (just for visualization purposes)
    rgb_values = np.zeros((reshaped_data.shape[0], 3))  # Fallback: black color for non-RGB data

# Create a figure with two subplots (for UMAP/t-SNE and RGB values)
fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# First subplot: UMAP or t-SNE projection colored by RGB values
axs[0].scatter(embedding[:, 0], embedding[:, 1], c=rgb_values, s=1, alpha=0.5)  # Scatter plot with RGB coloring
axs[0].set_title(f'{model_type.upper()} Projection of Raster Data')
axs[0].set_xlabel(f'{model_type.upper()} Component 1')
axs[0].set_ylabel(f'{model_type.upper()} Component 2')

# Second subplot: Display RGB values using imshow
# Reshape the RGB values back to (height, width, 3) for imshow
rgb_image = rgb_values.reshape(height, width, 3)  # Recreate the RGB image

# Show the RGB image using imshow
axs[1].imshow(rgb_image)
axs[1].axis('off')  # Hide axes for better visualization
axs[1].set_title('RGB Values of Raster')

# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show()

# Optionally, you can reshape the projection result to match the original raster shape (height, width, 2)
# umap_image = embedding.reshape(height, width, 2)

# If you want to save or process the UMAP/t-SNE projection back into image form, you can save it using rasterio or another library
