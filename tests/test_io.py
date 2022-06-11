#%%
from torchlight.utils.io import figsave, imread_float
from torchlight.utils.example import gray

import numpy as np

noisy = gray + 0.5 * np.random.randn(*gray.shape)

import matplotlib.pyplot as plt

plt.imsave('diff.png', np.abs(gray-noisy))
figsave('diff2.png', np.abs(gray-noisy))
# %%
