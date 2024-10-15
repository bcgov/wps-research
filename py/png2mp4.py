''' 20241015 convert png files to mp4 
'''
import os
import subprocess

# Get all PNG files in the current directory
png_files = [f for f in os.listdir('.') if f.endswith('.png')]
# Sort the files alphabetically
png_files.sort()

# Write the file names to filelist.txt
with open('.png2mp4_filelist.txt', 'w') as file:
    for png in png_files:
        file.write(f"file '{png}'\n")

# Run the ffmpeg command
ffmpeg_command = [
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', '.png2mp4_filelist.txt',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    'output.mp4'
]

subprocess.run(ffmpeg_command)
print("done")
