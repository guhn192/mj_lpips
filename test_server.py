# File: server.py
# pip install flask lpips pillow torchvision

import base64
import io
import json
import torch
import lpips
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
CORS(app)  # enable CORS for all routes and origins

# LPIPS model init (AlexNet)
lpips_model = lpips.LPIPS(net='alex')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips_model.to(device)

# transform to [-1,1]
transform_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

@app.route('/compute_lpips', methods=['POST'])
def compute_lpips():
    data = request.get_json()
    img1_b64 = data.get('img1')
    img2_b64 = data.get('img2')

    if not img1_b64 or not img2_b64:
        return jsonify({'error': 'Missing images'}), 400

    # decode base64 -> bytes
    img1_bytes = base64.b64decode(img1_b64.split(',')[1])  # remove "data:image/png;base64,"
    img2_bytes = base64.b64decode(img2_b64.split(',')[1])

    # Bytes -> Pillow Image
    img1_pil = Image.open(io.BytesIO(img1_bytes)).convert('RGB')
    img2_pil = Image.open(io.BytesIO(img2_bytes)).convert('RGB')

    # transform -> torch Tensor in [-1,1]
    img1_t = transform_lpips(img1_pil).unsqueeze(0).to(device)
    img2_t = transform_lpips(img2_pil).unsqueeze(0).to(device)

    # compute LPIPS
    dist = lpips_model(img1_t, img2_t)
    score = dist.item()

    return jsonify({'lpips_score': score})

if __name__ == '__main__':
    # e.g. python server.py
    app.run(host='0.0.0.0', port=5000, debug=True)
