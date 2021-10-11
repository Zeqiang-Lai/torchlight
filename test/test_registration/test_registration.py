# modified from https://github.com/quqixun/ImageRegistration


import numpy as np
from torchlight.utils.registration import Ransac
from torchlight.utils.registration import Align
from torchlight.utils.registration import Affine


# Affine Transform
# |x'|  = |a, b| * |x|  +  |tx|
# |y'|    |c, d|   |y|     |ty|
# pts_t =    A   * pts_s  + t

# -------------------------------------------------------------
# Test Class Affine
# -------------------------------------------------------------

# Create instance
af = Affine()

# Generate a test case as validation with
# a rate of outliers
outlier_rate = 0.9
A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate)

# At least 3 corresponding points to
# estimate affine transformation
K = 3
# Randomly select 3 pairs of points to do estimation
idx = np.random.randint(0, pts_s.shape[1], (K, 1))
A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

# Display known parameters with estimations
# They should be same when outlier_rate equals to 0,
# otherwise, they are totally different in some cases
print(A_true, '\n', t_true)
print(A_test, '\n', t_test)

# -------------------------------------------------------------
# Test Class Ransac
# -------------------------------------------------------------

# Create instance
rs = Ransac(K=3, threshold=1)

residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

# Run RANSAC to estimate affine tansformation when
# too many outliers in points set
A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
print(A_rsc, '\n', t_rsc)

# -------------------------------------------------------------
# Test Class Align
# -------------------------------------------------------------

# Load source image and target image
source_path = 'img1.png'
target_path = 'img2.png'

# Create instance
al = Align(threshold=1)

source = al.read_image(source_path)
target = al.read_image(target_path)

# Image transformation
warp = al.align_image(source_path, target_path)

# Merge warped image with target image to display
merge = np.uint8(np.hstack([source, warp, target]))

# Show the result
import cv2
cv2.imwrite('out.png', merge)

