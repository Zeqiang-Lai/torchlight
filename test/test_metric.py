import torch
from torchlight.utils.metrics import sam, psnr, ssim, mpsnr, mssim

a = torch.rand((16,3,32,32))
b = torch.rand((16,3,32,32))

print(sam(a,b))
print(psnr(a,b))
print(mpsnr(a,b))
print(mssim(a,b))
print(ssim(a,b))