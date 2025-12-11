import os
from PIL import Image

# ========
input_dir = r"/Users/yanyanru/desktop/genotype/train3/egrf"      
output_dir = r"/Users/yanyanru/desktop/genotype/train3/egrf2"    
TARGET_SIZE = (1360, 1024)                
background_color = (255, 255, 255)             
# =================================

os.makedirs(output_dir, exist_ok=True)

def resize_with_padding(img, target_size):
    """Scale the image proportionally and add borders around it to maintain its original shape without any distortion"""
    target_w, target_h = target_size
    w, h = img.size

    # Calculate the scaling to ensure that the image fits intact into the target canvas
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Scale proportionally
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create a new canvas with the desired target dimensions.
    new_img = Image.new("RGB", (target_w, target_h), background_color)

    # Centering the scaled image onto the canvas
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img

# Supported image file extensions
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for name in os.listdir(input_dir):
    ext = os.path.splitext(name)[1].lower()
    if ext not in exts:
        continue

    in_path = os.path.join(input_dir, name)
    out_path = os.path.join(output_dir, name)

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")              
            new_img = resize_with_padding(img, TARGET_SIZE)
            new_img.save(out_path, quality=95)
            print(f"Processing completed: {name}")
    except Exception as e:
        print(f"Processing {name} failed: {e}")
