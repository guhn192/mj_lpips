from PIL import Image
import torchvision.transforms as T
import torch
import lpips

# CPU 우선 사용 (안정성 확보)
device = torch.device("cpu")

# LPIPS 모델 로드
print("➡ 모델 로딩 중...")
model = lpips.LPIPS(net='vgg').to(device)
model.eval()
print("✅ 모델 로딩 완료")

# 이미지 경로 설정
img1_path = 'lpips/test/image1.jpg'
img2_path = 'lpips/test/image2.jpg'

# 이미지 로드 및 전처리
transform = T.Compose([
    T.Resize((64, 64)),  # 작게 리사이즈해서 빠르게 테스트
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

img1 = Image.open(img1_path).convert('RGB')
img2 = Image.open(img2_path).convert('RGB')

img1_t = transform(img1).unsqueeze(0).to(device)
img2_t = transform(img2).unsqueeze(0).to(device)

# LPIPS 거리 계산
print("➡ LPIPS 거리 계산 중...")
with torch.no_grad():
    dist = model(img1_t, img2_t)
print(f"✅ 계산 완료: image1 ↔ image2 LPIPS 거리 = {dist.item():.4f}")