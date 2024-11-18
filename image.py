import os
import shutil

# Paths to source and destination folders
source_folder = 'path_to_source_folder'
destination_folder = 'path_to_destination_folder'

# Supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get all image files in the source folder, sorted by their names
all_images = sorted([file for file in os.listdir(source_folder) if file.lower().endswith(image_extensions)])

# Take the last 617 images
images_to_move = all_images[-617:]

# Move images to the destination folder
for image in images_to_move:
    src_path = os.path.join(source_folder, image)
    dest_path = os.path.join(destination_folder, image)
    shutil.move(src_path, dest_path)

print(f"Moved {len(images_to_move)} images to '{destination_folder}'.")
