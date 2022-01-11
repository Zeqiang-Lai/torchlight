# test stateful transforms

from imageio import imread
import numpy as np

img = imread('tests/sample/lena.png')[:, :, :3]
img = img.astype('float') / 255


def test_rand_crop():
    from torchlight.transforms.stateful import RandCrop
    tsfm = RandCrop((50, 50))
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6

def test_flip_horizontal():
    from torchlight.transforms.stateful import FlipHorizontal
    tsfm = FlipHorizontal()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6
    
def test_flip_vertical():
    from torchlight.transforms.stateful import FlipVertical
    tsfm = FlipVertical()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6
    
def test_brightness_change():
    from torchlight.transforms.stateful import BrightnessChange
    tsfm = BrightnessChange()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6
    
def test_multiplicative_color_change():
    from torchlight.transforms.stateful import MultiplicativeColorChange
    tsfm = MultiplicativeColorChange()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6
    
def test_contrast():
    from torchlight.transforms.stateful import Contrast
    tsfm = Contrast()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6
    
def test_rotate():
    from torchlight.transforms.stateful import Rotate
    tsfm = Rotate()
    out1 = tsfm(img)
    out2 = tsfm(img)
    assert np.mean(np.abs(out1-out2)[:]) < 1e-6