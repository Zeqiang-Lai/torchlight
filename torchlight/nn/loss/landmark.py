from math import floor
import torch.nn as nn

# Unofficial Implementation of Landmark Loss in
# CrossNet++: Cross-scale Large-parallax Warping for Reference-based Super-resolution


def landmark_loss(landmarks, flow):
    """ 
        landmarks: cords of each paired landmarks [B,n,4]
        flow: offset of the flow [B,2,W,H]
    """
    # since landmark cord can be fractional, we need interpolation to
    # get the flow offset of each landmark.
    loss = 0
    B, _, W, H = flow.shape
    for i in range(B):
        for lm in landmarks[i]:
            x1, y1, x2, y2 = lm
            x1_d = floor(x1)
            y1_d = floor(y1)
            x1_u = x1_d+1
            y1_u = y1_d+1
            if x1_u >= W or y1_u >= H:
                continue
            a = flow[i, :, x1_d, y1_d]
            b = flow[i, :, x1_u, y1_u]
            c = flow[i, :, x1_u, y1_d]
            d = flow[i, :, x1_d, y1_u]
            o = a*(x1-x1_d)*(y1-y1_d) + b*(x1_u-x1)*(y1_u-x1) + c*(x1_u-x1)*(y1-y1_d) + d*(x1-x1_d)*(y1_u-x1)
            o_x, o_y = o[0], o[1]
            x1_warp = x1+o_x
            y1_warp = y1+o_y
            loss += (x1_warp-x2)**2 + (y1_warp-y2)**2
    loss = loss / (2*B)
    return loss


class LandmarkLoss(nn.Module):
    """ 
        landmarks: cords of each paired landmarks [B,n,4]
        flow: offset of the flow [B,2,W,H]
    """

    def forward(self, landmarks, flow):
        return landmark_loss(landmarks, flow)


def match_landmark_sift_knn_bbs(img1, img2):
    """ Match landmarks of two images with SIFT and KNN 

        Reference: https://github.com/ayushgarg31/Feature-Matching
    """

    import cv2

    t1 = cv2.imread(img1, 0)
    t2 = cv2.imread(img2, 0)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(t1, None)
    kp2, des2 = sift.detectAndCompute(t2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good1 = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)

    good2 = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good2.append([m])

    good = []

    for i in good1:
        img1_id1 = i[0].queryIdx
        img2_id1 = i[0].trainIdx

        (x1, y1) = kp1[img1_id1].pt
        (x2, y2) = kp2[img2_id1].pt

        for j in good2:
            img1_id2 = j[0].queryIdx
            img2_id2 = j[0].trainIdx

            (a1, b1) = kp2[img1_id2].pt
            (a2, b2) = kp1[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                good.append(i)

    match_points = []
    for g in good:
        img1_id1 = g[0].queryIdx
        img2_id1 = g[0].trainIdx
        (x1, y1) = kp1[img1_id1].pt
        (x2, y2) = kp2[img2_id1].pt

        match_points.append((x1, y1, x2, y2))

    visualize = cv2.drawMatchesKnn(t1, kp1, t2, kp2, good, None, [0, 0, 255], flags=2)

    return match_points, visualize
