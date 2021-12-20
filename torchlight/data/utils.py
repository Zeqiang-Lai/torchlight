import random


def get_patch(lr, hr, ref, patch_size=108, scale=2):

    L_h, L_w = lr.shape[:2]
    L_p = patch_size

    L_x = random.randrange(L_w // 4, 3 * L_w // 4 - L_p + 1 - 15)
    L_y = random.randrange(L_h // 4, 3 * L_h // 4 - L_p + 1 - 15)

    H_x, H_y = scale * L_x, scale * L_y
    H_p = scale * L_p

    patch_LR = lr[L_y : L_y + L_p, L_x : L_x + L_p, :]
    patch_HR = hr[H_y : H_y + H_p, H_x : H_x + H_p, :]
    delta = random.randint(0, 30)
    patch_ref = ref[
        (L_y - L_h // 4) * scale + delta : (L_y - L_h // 4) * scale + H_p + delta,
        (L_x - L_w // 4) * scale + delta : (L_x - L_w // 4) * scale + H_p + delta,
        :,
    ]
    if hr.shape == lr.shape:
        return patch_LR, patch_LR, patch_ref
    return patch_LR, patch_HR, patch_ref
