def visualize_gray(hsi):
    return hsi[:, 20, :, :]


def visualize_color(hsi):
    srf = tl.transforms.HSI2RGB().srf
    srf = torch.from_numpy(srf).float().to(hsi.device)
    hsi = hsi.permute(0, 2, 3, 1) @ srf.T
    return hsi.permute(0, 3, 1, 2)