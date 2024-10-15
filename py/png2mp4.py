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

# Determine the largest dimensions
max_width = 0
max_height = 0

for png in png_files:
    with Image.open(png) as img:
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)

print(f"Largest dimensions found: {max_width}x{max_height}")

# Resize images to match the largest dimensions and create filelist.txt
with open('filelist.txt', 'w') as file:
    for png in png_files:
        with Image.open(png) as img:
            # Resize the image
            resized_img = img.resize((max_width, max_height), Image.ANTIALIAS)
            resized_png = f"resized_{png}"  # New file name for the resized image
            resized_img.save(resized_png)  # Save the resized image
            file.write(f"file '{resized_png}'\n")

# Run the ffmpeg command with the resized images
ffmpeg_command = [
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', 'filelist.txt',
    '-vf', f'scale={max_width}:{max_height}',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    'output.mp4'
]

subprocess.run(ffmpeg_command)

# Optionally, clean up resized images after video creation
for png in png_files:
    resized_png = f"resized_{png}"
    if os.path.exists(resized_png):
        os.remove(resized_png)
