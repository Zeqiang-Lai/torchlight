from imageio import imread
import numpy as np

img = imread('tests/sample/lena.png')[:, :, :3]
img = img.astype('float') / 255

def test_mod_resize():
    from torchlight.transforms.functional import mod_resize
    for base in [5, 8, 16, 32]:
        img2 = mod_resize(img, base)
        assert img2.shape[0] % base == 0
        assert img2.shape[1] % base == 0