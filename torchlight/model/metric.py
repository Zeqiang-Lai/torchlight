import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# B, C, H, W

def psnr(X, Y):
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return peak_signal_noise_ratio(image_true=Y, image_test=X)

def ssim(X, Y):
    X = X.squeeze().detach().cpu().numpy()
    Y = Y.squeeze().detach().cpu().numpy()
    return structural_similarity(X, Y)