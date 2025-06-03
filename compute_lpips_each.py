import os
import csv
from PIL import Image
import torch
import lpips
from torchvision import transforms
from itertools import combinations
from collections import defaultdict

name = 'janggi_under6'
Direction = ['r', 'l', 'b']
BASE_DIR = f'../../dataset/{name}'


# === LPIPS model ===
lpips_model = lpips.LPIPS(net='vgg') # 네트워크가 vgg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips_model.to(device)

# === Transform ===
transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === 방향별 LPIPS 계산 ===
for direction in Direction:
    print(f"\n🔄 Processing direction: {direction.upper()}")
    
    IMAGE_DIR = os.path.join(BASE_DIR, f"{name}_{direction}")
    PAIRWISE_CSV = os.path.join(BASE_DIR, f"{name}_{direction}_lpips_results.csv")
    AVERAGE_CSV = os.path.join(BASE_DIR, f"{name}_{direction}_lpips_average_scores.csv")

    # === Load image filenames ===
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]

    # === Score Accumulator ===
    per_image_scores = {os.path.splitext(f)[0]: [] for f in image_files}
    total_score = 0
    count = 0
    
    # === Prepare Pairwise CSV and write header ===
    with open(PAIRWISE_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image1", "Image2", "LPIPS_Score"])

    # === Pairwise LPIPS computation ===
    for img1_path, img2_path in combinations(image_paths, 2):
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

        # Resize if necessary
            if img1.size != img2.size:
                target_size = (min(img1.width, img2.width), min(img1.height, img2.height))
                img1 = img1.resize(target_size)
                img2 = img2.resize(target_size)
                print("[Warning] img1 size and img2 size not matched!!")

            img1_t = transform_lpips(img1).unsqueeze(0).to(device)
            img2_t = transform_lpips(img2).unsqueeze(0).to(device)

            dist = lpips_model(img1_t, img2_t)
            score = dist.item()

            name1 = os.path.splitext(os.path.basename(img1_path))[0]
            name2 = os.path.splitext(os.path.basename(img2_path))[0]

            # Update scores
            per_image_scores[name1].append(score)
            per_image_scores[name2].append(score)
            total_score += score
            count += 1

            print(f"✓ {name1} vs {name2}: {score:.4f}")

            # === Append score to CSV immediately ===
            with open(PAIRWISE_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([name1, name2, score])
            #    f.flush()  # Ensure write is flushed to disk

        except Exception as e:
            print(f"⚠️ Failed to compute LPIPS for {img1_path} vs {img2_path}: {e}")

    # === Append average to Pairwise CSV ===
    with open(PAIRWISE_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Average LPIPS (All pairs)", "", total_score / count if count > 0 else 0.0])

    # === Save Per-Image Average CSV ===
    with open(AVERAGE_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Average LPIPS to Others"])

        per_image_avg_sum = 0
        for img_name, scores in per_image_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            writer.writerow([img_name, avg])
            per_image_avg_sum += avg

        overall_avg = per_image_avg_sum / len(image_files) if image_files else 0.0
        writer.writerow([])
        writer.writerow(["Overall Average Across Scenes", overall_avg])
    
    print(f"✅ Done with direction: {direction.upper()} (Total pairs: {count})")

# === 방향별 LPIPS 평균을 panoID 단위로 통합 ===
pano_scores_by_direction = defaultdict(dict)

for direction in Direction:
    avg_csv_path = os.path.join(BASE_DIR, f"{name}_{direction}_lpips_average_scores.csv")
    if not os.path.exists(avg_csv_path):
        print(f"⚠️ Missing average score file for direction {direction}")
        continue

    with open(avg_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or row[0].strip() == "" or "Overall Average" in row[0]:
                continue
            pano_id = row[0].strip()
            try:
                score = float(row[1])
                pano_scores_by_direction[pano_id][direction] = score
            except:
                print(f"⚠️ Could not parse score for {pano_id} in {direction}")

merged_csv_path = os.path.join(BASE_DIR, f"{name}_lpips_merged_average.csv")
with open(merged_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["panoID", "Average_LPIPS_Across_Directions", "f", "r", "l", "b"])

    for pano_id, dir_scores in pano_scores_by_direction.items():
        scores = [s for s in [dir_scores.get(d) for d in Direction] if s is not None]
        mean_score = sum(scores) / len(scores) if scores else ""
        writer.writerow([
            pano_id,
            mean_score,
            dir_scores.get('f', ''),
            dir_scores.get('r', ''),
            dir_scores.get('l', ''),
            dir_scores.get('b', '')
        ])

print(f"\n✅ Merged LPIPS averages saved to: {merged_csv_path}")

# print(f"\n✅ Done. Total pairs: {count}")
# print(f"📁 Pairwise CSV: {PAIRWISE_CSV}")
# print(f"📁 Per-Image Avg CSV: {AVERAGE_CSV}")
