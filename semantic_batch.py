import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# 설정
PROJECT_NAME = 'masan_100'
BASE_DIR = f'../../dataset/{PROJECT_NAME}'
DIRECTIONS = ['f', 'b', 'l', 'r']
SEGMENT_ROOT = os.path.join(BASE_DIR, f'{PROJECT_NAME}_f')
OUTPUT_CSV = os.path.join(BASE_DIR, f"{PROJECT_NAME}_ade20k_summary.csv")


# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
).to(device).eval()
extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

# Segformer 모델이 사용하는 전체 클래스 id ↔ label 매핑을 그대로 사용
TARGET_CLASSES = {
    name: idx for idx, name in model.config.id2label.items()
}

def segment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()


    # 🔁 segmentation 결과를 원본 크기로 정확히 리사이즈 (class ID 보존)
    seg_pil = Image.fromarray(seg.astype(np.int32), mode='I')
    seg_resized = seg_pil.resize(image.size, resample=Image.NEAREST)
    seg_resized = np.array(seg_resized)

    # 디버깅: 감지된 클래스 이름 출력
    detected_ids = np.unique(seg)
    print("📌 감지된 class IDs:", detected_ids)
    print("📌 감지된 class names:", [model.config.id2label[i] for i in detected_ids])

    total = seg_resized.size
    result = {}
    for name, class_id in TARGET_CLASSES.items():
        count = np.sum(seg_resized == class_id)
        result[name] = count / total
    return result

def process_segment(segment_id):
    ratios_list = []

    for d in DIRECTIONS:
        seg_dir = os.path.join(BASE_DIR, f'{PROJECT_NAME}_{d}', segment_id)
        if not os.path.exists(seg_dir):
            continue

        image_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(('.jpg', '.png'))])
        if len(image_files) == 0:
            continue

        img_path = os.path.join(seg_dir, image_files[0])
        try:
            ratios = segment_image(img_path)
            ratios_list.append(ratios)
        except Exception as e:
            print(f"⚠️ 오류: {img_path} - {e}")

    if len(ratios_list) == 0:
        return None

    avg = pd.DataFrame(ratios_list).mean().to_dict()
    avg['segment_ID'] = segment_id
    return avg

def run_all_segments():
    segment_list = sorted([
        d for d in os.listdir(SEGMENT_ROOT)
        if os.path.isdir(os.path.join(SEGMENT_ROOT, d))
    ])

    print(f"🔍 총 세그먼트 수: {len(segment_list)}")
    results = []

    for seg_id in tqdm(segment_list):
        row = process_segment(seg_id)
        if row:
            results.append(row)

    df = pd.DataFrame(results)
    df = df[['segment_ID'] + list(TARGET_CLASSES.keys())]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 저장 완료: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_all_segments()