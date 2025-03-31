import csv
from collections import defaultdict

# Input/output paths
INPUT_CSV = "../../dataset/seohyun/seohyun_lpips_results.csv"
OUTPUT_CSV = "../../dataset/seohyun/seohyun_lpips_average_scores.csv"

# Dictionary to store scores per image
image_scores = defaultdict(list)

# Read LPIPS result CSV
with open(INPUT_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or row[0].lower().startswith("average"):  # Skip summary
            continue
        if row[0] == "Image1":
            continue  # skip header

        img1, img2, score_str = row
        try:
            score = float(score_str)
            image_scores[img1].append(score)
            image_scores[img2].append(score)
        except ValueError:
            continue  # skip malformed rows

# Compute per-image and overall averages
per_image_averages = {}
total_score = 0
total_count = 0

for img, scores in image_scores.items():
    avg = sum(scores) / len(scores) if scores else 0.0
    per_image_averages[img] = avg
    total_score += avg
    total_count += 1

overall_avg = total_score / total_count if total_count > 0 else 0.0

# Write to output CSV
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Average LPIPS to Others"])

    for img, avg in sorted(per_image_averages.items()):
        writer.writerow([img, avg])

    writer.writerow([])
    writer.writerow(["Overall Average Across Scenes", overall_avg])

print(f"✅ Saved average results to {OUTPUT_CSV}")
