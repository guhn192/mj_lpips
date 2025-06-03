import os
import torch
import lpips
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from itertools import combinations
from tqdm import tqdm

# ========== 설정 ==========
PROJECT_NAME = '100_geomdan'
BASE_DIR = f'../../dataset/{PROJECT_NAME}'
OUTPUT_CSV = os.path.join(BASE_DIR, f"{PROJECT_NAME}_segment_lpips_summary.csv")
FOLDERS = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ========== LPIPS 모델 및 전처리 ==========
lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== LPIPS 계산 함수 ==========
def compute_lpips(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    if len(image_paths) < 2:
        return None

    total_score = 0
    count = 0
    for p1, p2 in combinations(image_paths, 2):
        try:
            img1 = Image.open(p1).convert("RGB")
            img2 = Image.open(p2).convert("RGB")
            if img1.size != img2.size:
                size = (min(img1.width, img2.width), min(img1.height, img2.height))
                img1 = img1.resize(size)
                img2 = img2.resize(size)
            img1_t = transform(img1).unsqueeze(0).to(device)
            img2_t = transform(img2).unsqueeze(0).to(device)
            with torch.no_grad():
                dist = lpips_model(img1_t, img2_t).item()
            total_score += dist
            count += 1
        except Exception as e:
            print(f"❌ Error processing {p1} vs {p2}: {e}")
            continue

    return total_score / count if count > 0 else None

# ========== 전체 폴더 반복 ==========
results = []
for folder in tqdm(FOLDERS, desc="Processing folders"):
    folder_path = os.path.join(BASE_DIR, folder)
    lpips_score = compute_lpips(folder_path)
    results.append({"segment_ID": folder, "Average_LPIPS": lpips_score})

# ========== 저장 ==========
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 저장 완료: {OUTPUT_CSV}")