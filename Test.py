import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
import cv2
from PIL import Image
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default=r"C:\Users\Administrator\Desktop\ploy2026\Polyp-PVT\PolypPVT.pth")
    opt = parser.parse_args()

    device = torch.device('cpu')

    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path, map_location=device))
    model.to(device)
    model.eval()

    image_root = r"D:\desktop\202508\消化道息肉检测\train\images"
    save_path = r"C:\Users\Administrator\Desktop\ploy2026\Polyp-PVT\test_ploy"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_list = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))]

    transform = transforms.Compose([
        transforms.Resize((opt.testsize, opt.testsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    for name in image_list:
        image_path = os.path.join(image_root, name)
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        h, w = img.size[1], img.size[0]
        image = transform(img).unsqueeze(0).to(device)
        P1, P2 = model(image)
        res = F.upsample(P1 + P2, size=(h, w), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        if name.lower().endswith('.jpg'):
            save_name = os.path.splitext(name)[0] + '.png'
        else:
            save_name = name
        cv2.imwrite(os.path.join(save_path, save_name), res * 255)
    print('Finish!')
