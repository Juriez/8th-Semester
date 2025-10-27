import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
from datetime import datetime

url = "https://upload.wikimedia.org/wikipedia/commons/7/7d/Dog_face.png"

req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
)
resp = urllib.request.urlopen(req)
image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

# custom convolution function
def custom_convolution(image, kernel):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

#  Choose kernel option

# Edge Detection
# kernel = np.array([
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1]
# ])
# kernel_name = "EdgeDetection"

#Identity
# kernel = np.array([
#     [0, 0, 0],
#     [0, 1, 0],
#     [0, 0, 0]
# ])
# kernel_name = "Identity"

# # Blur
# kernel = np.ones((3,3)) / 9.0
# kernel_name = "Blur"

# Sharpen
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
kernel_name = "Sharpen"

# Apply both convolutions
custom_output = custom_convolution(image, kernel)
opencv_output = cv2.filter2D(image, -1, kernel)

# output folder 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"output_{kernel_name}_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Save outputs
cv2.imwrite(os.path.join(output_folder, "original_image.png"), image)
cv2.imwrite(os.path.join(output_folder, "custom_output.png"), custom_output)
cv2.imwrite(os.path.join(output_folder, "opencv_output.png"), opencv_output)

# display comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Custom Convolution Output")
plt.imshow(custom_output, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("OpenCV filter2D Output")
plt.imshow(opencv_output, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\n Outputs saved in folder: '{output_folder}'")
