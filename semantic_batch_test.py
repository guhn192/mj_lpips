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
DIRECTIONS = ['f']
SEGMENT_ROOT = os.path.join(BASE_DIR, f'{PROJECT_NAME}_f')
OUTPUT_CSV = os.path.join(BASE_DIR, f"{PROJECT_NAME}_ade20k_summary.csv")

# 클래스 이름 (index → name)
ADE_CLASSES = {0: 'background', 1: 'wall', 2: 'building', 3: 'sky', 4: 'floor',
               7: 'road', 8: 'sidewalk', 9: 'earth', 10: 'tree', 11: 'plant', 13: 'car', 17: 'person'}

TARGET_CLASSES = {
    name: idx for idx, name in ADE_CLASSES.items()
    if name not in ['background', 'earth', 'floor']
}

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
).to(device).eval()
extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

def quick_test(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    detected_ids = np.unique(seg)

    print("📌 감지된 class IDs:", detected_ids)
    print("📌 감지된 class names:", [model.config.id2label[i] for i in detected_ids])
