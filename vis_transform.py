import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    RandAffined, RandFlipd, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd,
    RandZoomd, Compose, ToTensord
)
import torch
import random

# Set fixed seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Data config
root = "./data"
plane = "sagittal"
scan_id = "1207"
slice_idx = 16

# Load and prep image
img_path = os.path.join(root, "valid", plane, f"{scan_id}.npy")
img = np.load(img_path)[slice_idx]  # (256, 256)
img = np.expand_dims(img, axis=0)   # (1, 256, 256)
img = np.repeat(img, 3, axis=0)     # (3, 256, 256)
data_dict = {"image": img, "label": np.zeros(3)}  # dummy label

# Define transforms
transforms = [
    ("RandAffined", RandAffined(
        keys="image",
        prob=1.0,
        rotate_range=(0, 0, np.pi / 8),
        translate_range=(0.1, 0.1, 0.0),
        scale_range=(0.1, 0.1, 0.0),
        padding_mode="border"
    )),
    ("RandZoomd", RandZoomd(keys="image", prob=1.0, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", keep_size=True)),
    ("RandFlipd", RandFlipd(keys="image", spatial_axis=0, prob=1.0)),
    ("RandGaussianNoised", RandGaussianNoised(keys="image", prob=1.0, mean=0.0, std=0.1)),
    ("RandScaleIntensityd", RandScaleIntensityd(keys="image", prob=1.0, factors=0.2)),
    ("RandShiftIntensityd", RandShiftIntensityd(keys="image", prob=1.0, offsets=0.1)),
]

# Apply each transform and collect results
images = [img[0]]  # original image (only one channel shown)
titles = ["Original"]

for name, tfm in transforms:
    composed = Compose([tfm])
    result = composed(dict(image=np.copy(img), label=np.zeros(3)))
    images.append(result["image"][0])  # take channel 0 for grayscale display
    titles.append(name)

# Create grid: 2 rows, 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Plot each image
for i in range(len(images)):
    axes[i].imshow(images[i], cmap="gray")
    axes[i].set_title(titles[i])
    axes[i].axis("off")

# Hide extra subplot if needed
if len(images) < len(axes):
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

plt.tight_layout()
plt.savefig("vis_transform.png", dpi=500)
plt.show()