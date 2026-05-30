from __future__ import annotations

import math

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


def ensure_gray_uint8(image):
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    elif array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return array


def _crop_border(image, crop_border):
    if crop_border <= 0:
        return image
    return image[crop_border:-crop_border, crop_border:-crop_border]


def psnr_uint8(gt, pred, crop_border=0):
    gt = ensure_gray_uint8(gt)
    pred = ensure_gray_uint8(pred)
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt = _crop_border(gt, int(crop_border))
    pred = _crop_border(pred, int(crop_border))
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse <= 0:
        return float('inf')
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def ssim_uint8(gt, pred, crop_border=0):
    if not HAVE_SSIM:
        return None
    gt = ensure_gray_uint8(gt)
    pred = ensure_gray_uint8(pred)
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt = _crop_border(gt, int(crop_border))
    pred = _crop_border(pred, int(crop_border))
    try:
        return float(ssim_fn(gt, pred, data_range=255))
    except Exception:
        return None
