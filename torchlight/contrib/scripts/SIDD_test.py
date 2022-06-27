"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

from torchlight.utils.helper import load_checkpoint
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import cv2

from skimage import img_as_ubyte
import scipy.io as sio

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_dir', default='/media/exthdd/datasets/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='saved/SIDD_small_re/ckpt/model-epoch10.pth', type=str, help='Path to weights')
parser.add_argument('--save_img', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

result_dir = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir, exist_ok=True)

if args.save_img:
    result_dir_img = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_img, exist_ok=True)
    result_dir_img2 = os.path.join(args.result_dir, 'noisy')
    os.makedirs(result_dir_img2, exist_ok=True)
    
model_restoration = RSED()
load_checkpoint(model_restoration, args.weights)

print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration.eval()

# Process data
filepath = os.path.join(args.input_dir, 'BenchmarkNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['BenchmarkNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch[0],0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch
            if args.save_img:
                save_file = os.path.join(result_dir_img, '%04d_%02d.png'%(i+1,k+1))
                save_file2 = os.path.join(result_dir_img2, '%04d_%02d.png'%(i+1,k+1))
                cv2.imwrite(save_file, cv2.cvtColor(img_as_ubyte(restored_patch), cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_file2, cv2.cvtColor(img_as_ubyte(Inoisy[i,k,:,:,:]), cv2.COLOR_RGB2BGR))

restored = restored * 255
restored = restored.astype('uint8')
# save denoised data
sio.savemat(os.path.join(result_dir, 'Idenoised.mat'), {"Idenoised": restored,})
