# segment_lpips_batch.py (folder-based segment list version)

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import csv
import time
from PIL import Image
import torch
import lpips
from torchvision import transforms
from itertools import combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === 설정 ===
PROJECT_NAME = '100_geomdan'
BASE_DIR = f'../../dataset/{PROJECT_NAME}'
DIRECTIONS = ['f', 'r', 'l', 'b']
SEGMENT_ROOT = os.path.join(BASE_DIR, f'{PROJECT_NAME}_f')  # f 폴더 기준으로 segment 목록 추출
OUTPUT_CSV = os.path.join(BASE_DIR, f"{PROJECT_NAME}_segment_lpips_summary.csv")

# === LPIPS 모델 로딩 ===
lpips_model = lpips.LPIPS(net='vgg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips_model.to(device)
lpips_model.eval()

# === 이미지 전처리 ===
transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === 단일 segment 처리 ===
def process_segment(segment_ID):
    # LPIPS 모델을 각 프로세스 내에서 로드
    

    segment_result = {"segment_ID": segment_ID}

    for direction in DIRECTIONS:
        seg_dir = os.path.join(BASE_DIR, f"{PROJECT_NAME}_{direction}", segment_ID)
        if not os.path.exists(seg_dir):
            segment_result[direction] = None
            continue

        image_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(('.jpg', '.png'))])
        image_paths = [os.path.join(seg_dir, f) for f in image_files]

        if len(image_paths) < 2:
            segment_result[direction] = None
            continue

        total_score = 0
        count = 0

        for img1_path, img2_path in combinations(image_paths, 2):
            try:
                img1 = Image.open(img1_path).convert("RGB")
                img2 = Image.open(img2_path).convert("RGB")

                if img1.size != img2.size:
                    size = (min(img1.width, img2.width), min(img1.height, img2.height))
                    img1 = img1.resize(size)
                    img2 = img2.resize(size)

                img1_t = transform_lpips(img1).unsqueeze(0).to(device)
                img2_t = transform_lpips(img2).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist = lpips_model(img1_t, img2_t).item()

                total_score += dist
                count += 1
            except Exception as e:
                print(f"⚠️ 오류: {img1_path} vs {img2_path} - {e}")
                continue

        segment_result[direction] = total_score / count if count > 0 else None

    return segment_result

# === 전체 실행 ===
def run_all_segments():
    segment_list = sorted([
        d for d in os.listdir(SEGMENT_ROOT)
        if os.path.isdir(os.path.join(SEGMENT_ROOT, d))
    ])

    print(f"🔍 총 세그먼트 수: {len(segment_list)}")
    start_time = time.time()

    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(process_segment, segment_list), total=len(segment_list)))

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["segment_ID"] + DIRECTIONS + ["Average_LPIPS"])
        writer.writeheader()
        for row in results:
            valid_scores = [row[d] for d in DIRECTIONS if row[d] is not None]
            row["Average_LPIPS"] = sum(valid_scores) / len(valid_scores) if valid_scores else None
            writer.writerow(row)

    elapsed = time.time() - start_time
    print(f"\n⏱️ 분석 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")
    print(f"✅ LPIPS 세그먼트 요약 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_all_segments()
