"""Compare test metal.jpg with training metal images."""
from PIL import Image
import os
from pathlib import Path
import numpy as np

def analyze_image(path):
    img = Image.open(path)
    arr = np.array(img)
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    print(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")
    print(f"  Mean RGB: R={arr[:,:,0].mean():.1f}, G={arr[:,:,1].mean():.1f}, B={arr[:,:,2].mean():.1f}")
    print(f"  Brightness: {arr.mean():.1f}")
    print()

print("=" * 50)
print("TEST IMAGE: metal.jpg")
print("=" * 50)
analyze_image("Test_images/metal.jpg")

print("=" * 50)
print("TRAINING METAL IMAGES (sample)")
print("=" * 50)

train_metal = Path("data/processed/train/metal")
for img_path in list(train_metal.glob("*.jpg"))[:3]:
    print(f"Image: {img_path.name}")
    analyze_image(img_path)

print("=" * 50)
print("TRAINING PAPER IMAGES (sample) - for comparison")
print("=" * 50)

train_paper = Path("data/processed/train/paper")
for img_path in list(train_paper.glob("*.jpg"))[:3]:
    print(f"Image: {img_path.name}")
    analyze_image(img_path)
