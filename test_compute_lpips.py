
import os
import csv
from PIL import Image
import torch
import lpips
from torchvision import transforms
from itertools import combinations

# Folder containing merged panoramas
IMAGE_DIR = "../../dataset/pangyo/pangyo_merged"
CSV_OUTPUT = "../../dataset/pangyo/pangyo_lpips_results.csv"

# LPIPS model (AlexNet backbone)
lpips_model = lpips.LPIPS(net='alex')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips_model.to(device)

# Transform to [-1,1]
transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Get all image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]

# Prepare CSV
with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image1", "Image2", "LPIPS_Score"])

    total_score = 0
    count = 0

    for img1_path, img2_path in combinations(image_paths, 2):
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Resize to the same shape if necessary
            if img1.size != img2.size:
                target_size = (min(img1.width, img2.width), min(img1.height, img2.height))
                img1 = img1.resize(target_size)
                img2 = img2.resize(target_size)
                print("[Warning] img1 size and img2 size not matched!!")

            img1_t = transform_lpips(img1).unsqueeze(0).to(device)
            img2_t = transform_lpips(img2).unsqueeze(0).to(device)

            dist = lpips_model(img1_t, img2_t)
            score = dist.item()

            writer.writerow([os.path.basename(img1_path), os.path.basename(img2_path), score])
            total_score += score
            count += 1

            print(f"✓ {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}: {score:.4f}")

        except Exception as e:
            print(f"⚠️ Failed to compute LPIPS for {img1_path} vs {img2_path}: {e}")

    # Write average score
    avg_score = total_score / count if count > 0 else 0.0
    writer.writerow([])
    writer.writerow(["Average LPIPS", "", avg_score])
    print(f"\n✅ Done. Average LPIPS: {avg_score:.4f} | Pairs: {count}")