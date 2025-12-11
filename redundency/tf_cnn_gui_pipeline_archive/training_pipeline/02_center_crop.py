import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === 1. Input and output path ===
input_folder  = r"/Users/yanyanru/desktop/genotype/train3/egrf2"
output_folder = r"/Users/yanyanru/desktop/genotype/train3/egrf3"

os.makedirs(output_folder, exist_ok=True)

# === 2. Target size ===
TARGET_W = 1280   # Horizontal cropping
TARGET_H = 1024   # Maintain the original height

# === 3. Image list ===
images = []

# === 4. Loop through and process the images ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif"):
        img_path = os.path.join(input_folder, filename)

        # Reading the diagram（Grayscale）
        img = Image.open(img_path).convert("L")

        w, h = img.size  

        if (w, h) != (1360, 1024):
            print(f"The sizes don’t match, but the process is still being carried out: {filename}, size={img.size}")

        # === 5. Calculate the left and right cropping margins ===
        left   = (w - TARGET_W) // 2    
        right  = left + TARGET_W       
        top    = 0                   
        bottom = TARGET_H              

        img_cropped = img.crop((left, top, right, bottom))

        # Save the processed PNG
        save_path = os.path.join(output_folder, filename)
        img_cropped.save(save_path)

        # Save to numpy
        images.append(np.array(img_cropped))
        print(f"Already processed: {filename}")

# === 6. Compose as numpy array ===
images_np = np.stack(images, axis=0)
print("Final array shape:", images_np.shape)  # (N, 1024, 1280)

# === 7. save .npy ===
# np.save("C:/Users/GPU5/.../images_np1280x1024.npy", images_np)

# === 8. Display the first one ===
plt.imshow(images_np[0], cmap="gray")
plt.title("First Cropped Image")
plt.axis("off")
plt.show()
