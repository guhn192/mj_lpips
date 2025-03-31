import torch
import lpips
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# 1) LPIPS 모델 초기화 (AlexNet 기반)
loss_fn_alex = lpips.LPIPS(net='alex')  # net='vgg' 도 가능

# 2) 이미지를 불러올 때,
#    - PIL로 로드 → Tensor 변환
#    - [0,1] 범위를 [-1,1]로 정규화(Normalize(mean=0.5, std=0.5))
transform = transforms.Compose([
    transforms.ToTensor(),                   # [0,1] 범위의 FloatTensor (C,H,W)
    transforms.Normalize([0.5,0.5,0.5],      # mean
                         [0.5,0.5,0.5])      # std
])

# 3) 이미지 두 장 로딩
#    nvs_data/ 에 img0.png, img1.png 파일 있다고 가정
img0_path = "nvs_data/pano_tiles_512each_1.png"
img1_path = "nvs_data/pano_tiles_512each_4.png"

img0 = Image.open(img0_path).convert('RGB')
img1 = Image.open(img1_path).convert('RGB')

# 4) Tensor 변환 + 배치 차원 추가(1 x 3 x H x W)
img0_t = transform(img0).unsqueeze(0)
img1_t = transform(img1).unsqueeze(0)

print("Shape:", img0_t.shape)          # e.g. [1, 3, 512, 2048]
print("Min value:", img0_t.min().item())
print("Max value:", img0_t.max().item())

# 5) LPIPS distance 계산
#    결과는 perceptual distance를 나타내는 scalar 텐서
d = loss_fn_alex(img0_t, img1_t)

print("LPIPS distance:", d.item())



def unnormalize_and_save(img_t: torch.Tensor, save_path: str):
    """
    img_t: shape [1, 3, H, W], range [-1,1]
    save_path: e.g. 'debug_img.png'
    """
    # 1) Remove batch dimension -> shape [3,H,W]
    #    (assuming batch size is 1)
    img_3chw = img_t[0]  # shape [3,H,W]
    
    # 2) Un-normalize from [-1,1] back to [0,1]
    #    If you used mean=[0.5,...], std=[0.5,...], then to “undo”:
    #    x_un = (x_in * std) + mean, but for [0.5,0.5,0.5] that’s just *0.5 + 0.5
    img_3chw = (img_3chw * 0.5) + 0.5
    
    # 3) Clamp to [0,1] in case any values are slightly out of range
    img_3chw = img_3chw.clamp(0,1)
    
    # 4) Convert from torch.Tensor -> PIL Image
    #    Torch expects shape (C,H,W) in [0,1]
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(img_3chw)
    
    # 5) Save as PNG (or JPEG)
    pil_img.save(save_path)
    print(f"Saved debug image to {save_path}")

unnormalize_and_save(img0_t, "debug_img0.png")