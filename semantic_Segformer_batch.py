import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from scipy.stats import entropy

# ==========================
# ✅ 설정
# ==========================
PROJECT_NAME = 'masan_100'
BASE_DIR = f'../../dataset/{PROJECT_NAME}'
DIRECTIONS = ['f', 'b', 'l', 'r']
SEGMENT_ROOT = os.path.join(BASE_DIR, f'{PROJECT_NAME}_f')
OUTPUT_CSV = os.path.join(BASE_DIR, f"{PROJECT_NAME}_ade20k_summary.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# ✅ 모델: Segformer + ADE20K
# ==========================
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device).eval()
extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

# 포함할 클래스 및 그룹핑
INCLUDE_CLASSES = {
    'building': ['building', 'wall'],
    'car': ['car', 'truck', 'bus'],
    'sky': ['sky'],
    'tree': ['tree'],
    'road': ['road'],
    'grass': ['grass'],
    'sidewalk': ['sidewalk'],
    'person': ['person'],
    'plant': ['plant'],
    'fence': ['fence'],
    'railing': ['railing'],
    'signboard': ['signboard'],
    'streetlight': ['streetlight'],
    'pole': ['pole'],
    'awning': ['awning']
}

# 전체 라벨 매핑
ALL_LABELS = model.config.id2label
LABEL_TO_ID = {v: k for k, v in ALL_LABELS.items()}

# ==========================
# ✅ HSV 색상 클래스 정의
# ==========================
COLOR_CLASSES = {
    'red': [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],
    'yellow': [(20, 50, 50), (30, 255, 255)],
    'green': [(40, 50, 50), (70, 255, 255)],
    'cyan': [(80, 50, 50), (90, 255, 255)],
    'blue': [(100, 50, 50), (130, 255, 255)],
    'purple': [(140, 50, 50), (160, 255, 255)],
    'white': [(0, 0, 200), (180, 40, 255)],
    'black': [(0, 0, 0), (180, 255, 50)],
    'gray': [(0, 0, 50), (180, 40, 200)]
}

# ==========================
# ✅ Segmentation 및 분석 함수
# ==========================
def segment_ade20k(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    seg_pil = Image.fromarray(seg.astype(np.int32), mode='I')
    seg_resized = seg_pil.resize(image.size, resample=Image.NEAREST)
    seg_resized = np.array(seg_resized)
    total = seg_resized.size

    result = {}
    for new_class, original_classes in INCLUDE_CLASSES.items():
        ids = [LABEL_TO_ID[c] for c in original_classes if c in LABEL_TO_ID]
        count = sum(np.sum(seg_resized == class_id) for class_id in ids)
        result[new_class] = count / total
    return result

def segment_color(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    result = {}
    for name, bounds in COLOR_CLASSES.items():
        if name == 'red':
            lower1, upper1, lower2, upper2 = bounds
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = bounds
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = np.sum(mask > 0)
        result[name] = count / total_pixels
    return result

def compute_edge_entropy(image_path):
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    edges = cv2.Canny(image_np, 100, 200)
    hist, _ = np.histogram(edges, bins=2, range=(0, 256), density=True)
    return entropy(hist)

def compute_color_entropy(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=180, range=(0, 180), density=True)
    return entropy(h_hist)

def process_segment(segment_id):
    sem_ratios_list = []
    col_ratios_list = []
    edge_entropy_list = []
    color_entropy_list = []

    for d in DIRECTIONS:
        seg_dir = os.path.join(BASE_DIR, f'{PROJECT_NAME}_{d}', segment_id)
        if not os.path.exists(seg_dir):
            continue
        image_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(('.jpg', '.png'))])
        if len(image_files) == 0:
            continue
        img_path = os.path.join(seg_dir, image_files[0])
        try:
            sem = segment_ade20k(img_path)
            col = segment_color(img_path)
            edge_ent = compute_edge_entropy(img_path)
            color_ent = compute_color_entropy(img_path)
            sem_ratios_list.append(sem)
            col_ratios_list.append(col)
            edge_entropy_list.append(edge_ent)
            color_entropy_list.append(color_ent)
        except Exception as e:
            print(f"⚠️ 오류: {img_path} - {e}")

    if len(sem_ratios_list) == 0 or len(col_ratios_list) == 0:
        return None

    sem_avg = pd.DataFrame(sem_ratios_list).mean().to_dict()
    col_avg = pd.DataFrame(col_ratios_list).mean().to_dict()
    merged = {**sem_avg, **col_avg}
    merged['segment_ID'] = segment_id
    merged['visual_complexity'] = np.mean(edge_entropy_list)
    merged['color_complexity'] = np.mean(color_entropy_list)
    return merged

def run_all_segments():
    segment_list = sorted([d for d in os.listdir(SEGMENT_ROOT) if os.path.isdir(os.path.join(SEGMENT_ROOT, d))])
    results = []
    for seg_id in tqdm(segment_list):
        row = process_segment(seg_id)
        if row:
            results.append(row)
    df = pd.DataFrame(results)
    col_order = ['segment_ID'] + list(INCLUDE_CLASSES.keys()) + list(COLOR_CLASSES.keys()) + ['visual_complexity', 'color_complexity']
    df = df[col_order]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 저장 완료: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_all_segments()
