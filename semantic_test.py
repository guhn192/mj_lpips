import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
extractor = SegformerFeatureExtractor.from_pretrained(model_name)

# 클래스 색상 맵
def get_ade20k_colormap(num_classes=150):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

COLORMAP = get_ade20k_colormap()
id2label = model.config.id2label  # class ID → 이름 매핑

def visualize_segmentation(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    color_seg = COLORMAP[seg]
    color_seg = Image.fromarray(color_seg.astype(np.uint8))
    return image, color_seg, seg

# 실행 파트
DIRECTIONS = ['f', 'b', 'l', 'r']
segment_id = '93'
base_path = '../../dataset/test'

for direction in DIRECTIONS:
    dir_path = os.path.join(base_path, f'test_{direction}', segment_id)
    save_dir = os.path.join(base_path, f'seg_{direction}')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📂 분석 중: {direction.upper()} 방향")
    if not os.path.exists(dir_path):
        print(f"❌ 경로 없음: {dir_path}")
        continue

    image_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))])
    if not image_files:
        print("⚠️ 이미지 없음")
        continue

    for file in image_files:
        img_path = os.path.join(dir_path, file)
        print(f"🔍 {file}")
        image, color_seg, seg = visualize_segmentation(img_path)

        # 사용된 클래스 추출 (중복 제거)
        unique_classes = np.unique(seg)
        legend_elements = []
        for class_id in unique_classes:
            if class_id in id2label:
                class_name = id2label[class_id]
                color = COLORMAP[class_id] / 255.0  # matplotlib에 맞게 정규화
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=class_name))

        # 시각화 및 저장
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[1].imshow(color_seg)
        axes[1].set_title("Segmentation")
        axes[1].legend(handles=legend_elements, loc='lower left', fontsize='small', framealpha=0.7)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()

        file_base = os.path.splitext(file)[0]
        save_path = os.path.join(save_dir, f"seg_{direction}_{file_base}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 저장됨: {save_path}")