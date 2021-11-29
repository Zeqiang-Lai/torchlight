from torchlight.metrics import psnr, mpsnr, mssim, sam, set_data_format, FORMAT_HWC, FORMAT_CHW
from imageio import imread, imwrite
import numpy as np

set_data_format(FORMAT_CHW)

x = np.random.rand(16, 3, 512, 512)
y = np.random.rand(16, 3, 512, 512)

print(psnr(x, y))

set_data_format(FORMAT_HWC)

img1 = imread('demo/lena.png')[:, :, :3]
img1 = img1.astype('float') / 255
noise = np.random.randn(*img1.shape) * 30 / 255
img2 = img1+noise
img2 = img2.clip(0,1)
imwrite('noise.png', (img2*255).astype('uint8'))
print(mpsnr(img1, img2))
print(psnr(img1, img2))
print(sam(img1, img2))