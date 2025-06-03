# ✅ Segmentation 및 복잡도 계산 코드 (multi_test 하위 A~I 폴더 대상)

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from scipy.stats import entropy

# ===== 설정 =====
BASE_DIR = '../../dataset/multi_test'
OUTPUT_CSV = os.path.join(BASE_DIR, 'segmentation_complexity_summary.csv')
FOLDERS = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===== Segformer 모델 로딩 =====
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device).eval()
extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
LABEL_TO_ID = {v: k for k, v in model.config.id2label.items()}
SKY_ID = LABEL_TO_ID['sky']

# ===== 계산 함수 정의 =====
def compute_edge_entropy(image_path):
    gray = np.array(Image.open(image_path).convert("L"))
    edges = cv2.Canny(gray, 100, 200)
    hist, _ = np.histogram(edges, bins=2, range=(0, 256), density=True)
    return entropy(hist)

def compute_color_entropy(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=180, range=(0, 180), density=True)
    return entropy(h_hist)

def compute_openness(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    seg_resized = Image.fromarray(seg.astype(np.int32), mode='I').resize(image.size, resample=Image.NEAREST)
    seg_np = np.array(seg_resized)
    return np.sum(seg_np == SKY_ID) / seg_np.size

# ===== 전체 폴더 반복 실행 =====
results = []
for folder in tqdm(FOLDERS):
    folder_path = os.path.join(BASE_DIR, folder)
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    edge_entropies = [compute_edge_entropy(p) for p in image_files]
    color_entropies = [compute_color_entropy(p) for p in image_files]
    openness_vals = [compute_openness(p) for p in image_files]

    results.append({
        "segment_ID": folder,
        "visual_complexity": np.mean(edge_entropies),
        "color_complexity": np.mean(color_entropies),
        "openness": np.mean(openness_vals)
    })

# ===== 저장 =====
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 저장 완료: {OUTPUT_CSV}")