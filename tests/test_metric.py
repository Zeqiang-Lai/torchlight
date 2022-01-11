from torchlight.metrics import psnr, ssim, mpsnr, mssim, set_data_format, FORMAT_HWC, FORMAT_CHW
from imageio import imread
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

eps = 1e-8

img1 = imread('tests/sample/lena.png')[:, :, :3]
img1 = img1.astype('float') / 255
noise = np.random.randn(*img1.shape) * 30 / 255
img2 = img1+noise
img2 = img2.clip(0, 1)

gt_psnr = peak_signal_noise_ratio(img1, img2)
gt_ssim = structural_similarity(img1, img2, channel_axis=2)

gt_mpsnr = 0
for i in range(3):
    gt_mpsnr += peak_signal_noise_ratio(img1[:,:,i], img2[:,:,i])
gt_mpsnr /= 3


class TestPSNR:
    def test_basic(self):
        set_data_format(FORMAT_HWC)
        ours = psnr(img1, img2)
        assert abs(gt_psnr-ours) < eps

    def test_torch(self):
        import torch
        set_data_format(FORMAT_HWC)
        img1_torch = torch.from_numpy(img1)
        img2_torch = torch.from_numpy(img2)
        ours = psnr(img1_torch, img2_torch)
        assert abs(gt_psnr-ours) < eps

    def test_chw(self):
        set_data_format(FORMAT_CHW)
        ours = psnr(img1.transpose(2, 0, 1),
                    img2.transpose(2, 0, 1))
        assert abs(gt_psnr-ours) < eps

    def test_batch(self):
        set_data_format(FORMAT_HWC)
        ours = psnr(img1[None],
                    img2[None])
        assert abs(gt_psnr-ours) < eps


class TestSSIM:
    def test_basic(self):
        set_data_format(FORMAT_HWC)
        ours = ssim(img1, img2)
        assert abs(gt_ssim-ours) < eps

    def test_torch(self):
        import torch
        set_data_format(FORMAT_HWC)
        img1_torch = torch.from_numpy(img1)
        img2_torch = torch.from_numpy(img2)
        ours = ssim(img1_torch, img2_torch)
        assert abs(gt_ssim-ours) < eps

    def test_chw(self):
        set_data_format(FORMAT_CHW)
        ours = ssim(img1.transpose(2, 0, 1),
                    img2.transpose(2, 0, 1))
        assert abs(gt_ssim-ours) < eps

    def test_batch(self):
        set_data_format(FORMAT_HWC)
        ours = ssim(img1[None],
                    img2[None])
        assert abs(gt_ssim-ours) < eps


class TestMSSIM:
    def test_basic(self):
        set_data_format(FORMAT_HWC)
        ours = mssim(img1, img2)
        assert abs(gt_ssim-ours) < eps

    def test_torch(self):
        import torch
        set_data_format(FORMAT_HWC)
        img1_torch = torch.from_numpy(img1)
        img2_torch = torch.from_numpy(img2)
        ours = mssim(img1_torch, img2_torch)
        assert abs(gt_ssim-ours) < eps

    def test_chw(self):
        set_data_format(FORMAT_CHW)
        ours = mssim(img1.transpose(2, 0, 1),
                     img2.transpose(2, 0, 1))
        assert abs(gt_ssim-ours) < eps

    def test_batch(self):
        set_data_format(FORMAT_HWC)
        ours = mssim(img1[None],
                     img2[None])
        assert abs(gt_ssim-ours) < eps


class TestMPSNR:
    def test_basic(self):
        set_data_format(FORMAT_HWC)
        ours = mpsnr(img1, img2)
        assert abs(gt_mpsnr-ours) < eps

    def test_torch(self):
        import torch
        set_data_format(FORMAT_HWC)
        img1_torch = torch.from_numpy(img1)
        img2_torch = torch.from_numpy(img2)
        ours = mpsnr(img1_torch, img2_torch)
        assert abs(gt_mpsnr-ours) < eps

    def test_chw(self):
        set_data_format(FORMAT_CHW)
        ours = mpsnr(img1.transpose(2, 0, 1),
                    img2.transpose(2, 0, 1))
        assert abs(gt_mpsnr-ours) < eps

    def test_batch(self):
        set_data_format(FORMAT_HWC)
        ours = mpsnr(img1[None],
                    img2[None])
        assert abs(gt_mpsnr-ours) < eps
