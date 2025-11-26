import os
import random
import shutil
import glob

src = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_magnification_cropped_png"
dst = "/home/clab/Downloads/jiaxin_temporal/Droso/data/40X_sampled_png"

os.makedirs(dst, exist_ok=True)

# 找到所有 png 图
paths = glob.glob(os.path.join(src, "*.png"))
print("Total images:", len(paths))

# 抽取 1/10
k = max(1, len(paths) // 20)
sampled = random.sample(paths, k)

print("Sampled:", len(sampled))

# 复制文件
for p in sampled:
    shutil.copy(p, dst)

print("Done!")
