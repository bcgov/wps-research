'''20250219 convert png files in present folder to video ( output.mp4 ) 
'''
import cv2
import os

image_folder = './'
output_video = 'output.mp4'
frame_rate = 30  # Frames per second

images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort()  # Sort the images by name

# Get the dimensions of the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v' for MP4
out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Write the images to the video
for image in images:
    print(image)
    img = cv2.imread(os.path.join(image_folder, image))
    out.write(img)

out.release()
print("Video saved to", output_video)


