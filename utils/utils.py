import numpy as np
import matplotlib.pyplot as plt


def get_subsets(A, B, N: int, randomize: bool = False):
    dset_length = A.shape[0]

    if randomize:
        idx = np.random.permutation(dset_length)
        return (A[idx[:N], :], B[idx[:N], :])
    else:
        return (A[:N, :], B[:N, :])


def select_pixels(img, idxs):
    idxs = idxs.astype(int)
    X, Y = np.meshgrid(idxs, idxs)
    img = img[X.flatten(), Y.flatten()]
    img = np.reshape(img, (len(idxs), len(idxs)))
    return img


def crop_img(img, crop: int = 3):
    return img[crop:-crop, crop:-crop]

