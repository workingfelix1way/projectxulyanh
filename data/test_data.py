import os
from PIL import Image

data_dir = "./data/kaggle_3m/"
valid_images = 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.tif'):
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img.verify()
                valid_images += 1
            except Exception as e:
                print(f"Error: {file} - {e}")

print(f"Valid images: {valid_images}")