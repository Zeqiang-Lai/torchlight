#%%
from torchlight.transforms.functional import imresize
from torchlight.utils.io import imread_uint, imsave
from scipy.misc import face

import matplotlib.pyplot as plt

img = face()
imsave('val.png', img)

img = imread_uint('val.png')
plt.imshow(img)

img = imresize(img, height=150)
imsave('val2.png', img)
# %%
