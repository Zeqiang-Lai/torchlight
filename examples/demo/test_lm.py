#%%
from torchlight.nn.loss.landmark import match_landmark_sift_knn_bbs
import matplotlib.pyplot as plt

matches, visualize = match_landmark_sift_knn_bbs('lena.png', 'lena.png')
plt.imshow(visualize, interpolation = 'bicubic')
plt.axis('off')
plt.show()
# %%
