import cv2
import numpy as np
from skimage.feature import hog


def compute_ITA(img):
    """
    Takes LAB image and we cmpute the ITA Value
    """
    # lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # plt.imshow(lab_img)

    L, a, b = cv2.split(img)
    # L = cv2.equalizeHist(L)
    # print("shape L {} shape b {}".format(L.shape, b.shape))
    # print(L, b)
    L = L.reshape(img.shape[0] * img.shape[1])
    b = b.reshape(img.shape[0] * img.shape[1])

    L_mean = np.mean(L)
    b_mean = np.mean(b)

    # print("Lmean: ",L_mean)
    # print("bmean: ", b_mean)

    L_sd = np.std(L, axis=0)  # .reshape(256, -1)
    b_sd = np.std(b, axis=0)  # .reshape(256, -1)

    L = L[np.where(L > (L_mean - (1 * L_sd)))]
    L = L[np.where(L < (L_mean + (1 * L_sd)))]

    b = b[np.where(b > (b_mean - (1 * b_sd)))]
    b = b[np.where(b < (b_mean + (1 * b_sd)))]

    L_mean = np.mean(L)
    b_mean = np.mean(b)

    ita = (np.arctan2((L_mean - 50), b_mean)) * (180 / np.pi)

    # print(ita)
    return L_mean, b_mean, L_sd, b_sd, ita


def preprocesing_masks_for_classification(masks, size=(256, 256), ita=True):
    """Return raw pixel information and
    feature based on hog and mean-std"""
    cielab_pixels = list()
    rgb_pixels = list()
    features = list()

    for m in masks:
        tmp = cv2.resize(m, size)
        tmp_lab = cv2.cvtColor(tmp, cv2.COLOR_BGR2LAB)
        color_feats = np.concatenate(
            [
                [np.mean(tmp_lab[:, :, i]), np.std(tmp_lab[:, :, i])]
                for i in range(tmp_lab.shape[2])
            ]
        )

        hog_feats = hog(
            tmp_lab,
            orientations=32,
            pixels_per_cell=size,
            cells_per_block=(1, 1),
            visualize=False,
            multichannel=True,
        )
        if ita:
            L_mean, b_mean, L_sd, b_sd, ita = compute_ITA(tmp_lab)
            features.append(
                np.concatenate(
                    (hog_feats, color_feats, [L_mean], [b_mean], [L_sd], [b_sd], [ita])
                )
            )
        else:
            features.append(np.concatenate((hog_feats, color_feats)))

        rgb_pixels.append(tmp.reshape(-1, 1).ravel())
        cielab_pixels.append(tmp_lab.reshape(-1, 1).ravel())

    return cielab_pixels, rgb_pixels, features
