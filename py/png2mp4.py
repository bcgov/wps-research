''' 20241015 convert png files to mp4 
'''
import os
import subprocess
from PIL import Image

# Get all PNG files in the current directory
png_files = [f for f in os.listdir('.') if f.endswith('.png')]
# Sort the files alphabetically
png_files.sort()

if not png_files:
    print("No PNG files found in the current directory.")
    exit(1)

# Check dimensions of the PNG files
dimensions = None

for png in png_files:
    with Image.open(png) as img:
        if dimensions is None:
            dimensions = img.size  # (width, height)
        elif dimensions != img.size:
            print(f"Dimension mismatch: {png} has size {img.size}, expected {dimensions}.")
            exit(1)

print(f"All images have the same dimensions: {dimensions}")

# Write the file names to filelist.txt
with open('.png2mp4_filelist.txt', 'w') as file:
    for png in png_files:
        file.write(f"file '{png}'\n")

# Use the common dimensions
width, height = dimensions

# Run the ffmpeg command with scaling
ffmpeg_command = [
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', '.png2mp4_filelist.txt',
    '-vf', f'scale={width}:{height}',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    'output.mp4'
]

subprocess.run(ffmpeg_command)

