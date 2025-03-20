import cv2
import os
import matplotlib.pyplot as plt

data_dir = "dataset/handloom/"
img_name = os.listdir(data_dir)[0]  # Load first image
img_path = os.path.join(data_dir, img_name)

# Read and display image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title("Sample Saree Image")
plt.show()
