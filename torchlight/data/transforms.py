import torch
import numpy as np

class GaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        img_L = img + torch.normal(0, self.sigma, img.shape)
        return img_L