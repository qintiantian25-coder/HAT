#!/usr/bin/env python3
"""Evaluate full-image and blind-pixel metrics for model outputs.

Usage:
  python tools/evaluate_blind.py --out_dir /path/to/outputs --gt_dir /path/to/gts \
    --mask_csv /path/to/test_mask/001/blind_pixel_coords.csv --save_dir /path/to/save
"""
import os
import re
import csv
import argparse
import numpy as np
import cv2
from math import log10

try:
    from skimage.metrics import structural_similarity as ssim_fn
    have_ssim = True
except Exception:
    have_ssim = False


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def psnr_uint8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * log10((255.0 ** 2) / mse)


def compute_ssim(a, b):
    if not have_ssim:
        return None
    try:
        s = ssim_fn(a, b, data_range=255)
        return float(s)
    except Exception:
        return None


def evaluate(out_dir, gt_dir, input_dir=None, mask_csv=None, save_dir=None):
    out_imgs = sorted([f for f in os.listdir(out_dir) if f.lower().endswith('.png')], key=natural_sort_key)
    gt_map = {}
    for root, _, files in os.walk(gt_dir):
        for f in files:
            if f.lower().endswith('.png'):
                gt_map[f] = os.path.join(root, f)

    input_map = {}
    if input_dir and os.path.exists(input_dir):
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith('.png'):
                    input_map[f] = os.path.join(root, f)

    blind_coords = load_blind_coords(mask_csv)

    per_image_logs = []
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0

    print(f"===> Evaluating {len(out_imgs)} output images...")
    for img_name in out_imgs:
        out_path = os.path.join(out_dir, img_name)
        gt_path = gt_map.get(img_name)
        if gt_path and os.path.exists(out_path):
            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is None or out_img is None:
                continue
            if out_img.shape != gt_img.shape:
                out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

            full_psnr = psnr_uint8(out_img, gt_img)
            full_ssim = compute_ssim(out_img, gt_img)

            row = {
                'image': img_name,
                'psnr': float(full_psnr),
                'ssim': float(full_ssim) if full_ssim is not None else None,
                'blind_mae': None,
                'blind_rmse': None,
                'blind_psnr': None,
                'blind_mae_input': None,
                'blind_mae_gain_abs': None,
                'blind_mae_gain_pct': None,
                'blind_count': 0
            }

            if blind_coords is not None:
                h, w = gt_img.shape[:2]
                x = blind_coords[:, 0]
                y = blind_coords[:, 1]
                valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                if np.any(valid):
                    x = x[valid]
                    y = y[valid]
                    gt_vals = gt_img[y, x].astype(np.float64)
                    out_vals = out_img[y, x].astype(np.float64)
                    err = out_vals - gt_vals

                    blind_abs = np.abs(err)
                    blind_sq = err ** 2

                    blind_abs_sum += float(blind_abs.sum())
                    blind_sq_sum += float(blind_sq.sum())
                    blind_pix_sum += int(len(err))

                    in_path = input_map.get(img_name)
                    in_mae = None
                    if in_path and os.path.exists(in_path):
                        in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                        if in_img is not None:
                            if in_img.shape != gt_img.shape:
                                in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                            in_vals = in_img[y, x].astype(np.float64)
                            in_err = in_vals - gt_vals
                            in_abs = np.abs(in_err)
                            in_sq = in_err ** 2
                            blind_abs_in_sum += float(in_abs.sum())
                            blind_sq_in_sum += float(in_sq.sum())
                            in_mae = float(in_abs.mean())

                    row.update({
                        'blind_mae': float(blind_abs.mean()),
                        'blind_rmse': float(np.sqrt(blind_sq.mean())),
                        'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                        'blind_mae_input': in_mae,
                        'blind_count': int(len(err))
                    })
                    if in_mae is not None:
                        row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                        row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

            per_image_logs.append(row)

    # print summary
    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))
        print("===> Blind-Pixel Focused Metrics")
        print(f"BlindCount(total sampled): {blind_pix_sum}")
        print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}")

    # save per-image CSV
    if save_dir is None:
        save_dir = os.path.join(out_dir, '..', 'eval')
    os.makedirs(save_dir, exist_ok=True)
    save_csv = os.path.join(save_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = [
            'image', 'psnr', 'ssim',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
        ]
        with open(save_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_logs:
                writer.writerow(row)
        print(f"Per-image test metrics saved to: {save_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True, help='Directory with model output PNGs')
    p.add_argument('--gt_dir', required=True, help='Ground-truth images directory')
    p.add_argument('--input_dir', default=None, help='Input LR images dir (for input blind error)')
    p.add_argument('--mask_csv', default=None, help='CSV with blind pixel coords')
    p.add_argument('--save_dir', default=None, help='Directory to save per-image metrics CSV')
    args = p.parse_args()

    evaluate(args.out_dir, args.gt_dir, args.input_dir, args.mask_csv, args.save_dir)


if __name__ == '__main__':
    main()
"""Evaluate full-frame and blind-pixel metrics for a set of output images.

Usage example:
  python tools/evaluate_blind.py \
    --dataset_path /home/student_server/Qtt/NAFNet/data \
    --output_dir /path/to/model_outputs \
    --save_dir ./results/eval

The script expects the dataset layout described by the user:
  data/
    test_blur/
    test_sharp/  (ground truth)
    test_mask/   (optional CSV coords under subfolders)

It will compute per-image PSNR/SSIM and blind-region metrics (MAE, RMSE, PSNR)
and save a CSV with per-image rows and print aggregated blind metrics.
"""
import argparse
import os
import os.path as osp
import re
import csv
import numpy as np
import cv2
from math import log10

try:
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    compare_ssim = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    if not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def psnr_from_mse(mse):
    if mse <= 0:
        return float('inf')
    return 10.0 * log10((255.0 * 255.0) / mse)


def calc_psnr(gt, out):
    gt = gt.astype(np.float64)
    out = out.astype(np.float64)
    mse = np.mean((gt - out) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * log10((255.0 * 255.0) / mse)


def calc_ssim(gt, out):
    if compare_ssim is None:
        raise RuntimeError('skimage is required for SSIM. pip install scikit-image')
    # skimage expects images in 2D for grayscale
    s = compare_ssim(gt, out, data_range=255)
    return float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Absolute path to data folder')
    parser.add_argument('--output_dir', required=True, help='Directory with model outputs (.png)')
    parser.add_argument('--save_dir', default='./results/eval', help='Where to save metrics CSV')
    parser.add_argument('--test_mask_csv', default=None, help='Optional explicit CSV with blind coords')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_dir = args.output_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # determine blind coords file
    if args.test_mask_csv is not None:
        test_mask_csv = args.test_mask_csv
    else:
        test_mask_csv = osp.join(dataset_path, 'test_mask', '001', 'blind_pixel_coords.csv')

    blind_coords = load_blind_coords(test_mask_csv)

    # build maps for GT and input (blur)
    gt_root = osp.join(dataset_path, 'test_sharp')
    input_root = osp.join(dataset_path, 'test_blur')

    gt_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                gt_map[f] = osp.join(root, f)

    input_map = {}
    if os.path.exists(input_root):
        for root, _, files in os.walk(input_root):
            for f in files:
                if f.lower().endswith('.png'):
                    input_map[f] = osp.join(root, f)

    out_imgs = sorted([f for f in os.listdir(output_dir) if f.lower().endswith('.png')], key=natural_sort_key)

    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []

    print(f"===> 开始定量打分，准备比对 {len(out_imgs)} 张图片...")
    for img_name in out_imgs:
        out_path = osp.join(output_dir, img_name)
        gt_path = gt_map.get(img_name)
        if gt_path and os.path.exists(out_path):
            out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None and out_img is not None:
                if out_img.shape != gt_img.shape:
                    out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

                try:
                    full_psnr = calc_psnr(gt_img, out_img)
                except Exception:
                    full_psnr = None
                try:
                    full_ssim = calc_ssim(gt_img, out_img) if compare_ssim is not None else None
                except Exception:
                    full_ssim = None

                row = {
                    'image': img_name,
                    'psnr': full_psnr,
                    'ssim': full_ssim,
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'blind_mae_input': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                    'blind_count': 0
                }

                if blind_coords is not None:
                    h, w = gt_img.shape[:2]
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]

                        gt_vals = gt_img[y, x].astype(np.float64)
                        out_vals = out_img[y, x].astype(np.float64)
                        err = out_vals - gt_vals

                        blind_abs = np.abs(err)
                        blind_sq = err ** 2

                        blind_abs_sum += float(blind_abs.sum())
                        blind_sq_sum += float(blind_sq.sum())
                        blind_pix_sum += int(len(err))

                        # input image (blur)
                        in_path = input_map.get(img_name)
                        in_mae = None
                        if in_path and os.path.exists(in_path):
                            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                            if in_img is not None:
                                if in_img.shape != gt_img.shape:
                                    in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                                in_vals = in_img[y, x].astype(np.float64)
                                in_err = in_vals - gt_vals
                                in_abs = np.abs(in_err)
                                in_sq = in_err ** 2
                                blind_abs_in_sum += float(in_abs.sum())
                                blind_sq_in_sum += float(in_sq.sum())
                                in_mae = float(in_abs.mean())

                        row.update({
                            'blind_mae': float(blind_abs.mean()),
                            'blind_rmse': float(np.sqrt(blind_sq.mean())),
                            'blind_psnr': float(psnr_from_mse(float(blind_sq.mean()))),
                            'blind_mae_input': in_mae,
                            'blind_count': int(len(err))
                        })
                        if in_mae is not None:
                            row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                            row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

                per_image_logs.append(row)

    # save per-image CSV
    save_blind_dir = osp.join(save_dir, 'blind_eval')
    os.makedirs(save_blind_dir, exist_ok=True)
    save_blind_csv = osp.join(save_blind_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = [
            'image', 'psnr', 'ssim',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
        ]
        with open(save_blind_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_logs:
                writer.writerow(row)
        print(f"Per-image test metrics saved to: {save_blind_csv}")

    # aggregate blind metrics
    if blind_coords is not None and blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(psnr_from_mse(blind_mse))

        print("===> Blind-Pixel Focused Metrics")
        print(f"BlindCoordsCSV: {test_mask_csv}")
        print(f"BlindCount(total sampled): {blind_pix_sum}")
        print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}")

        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = float(psnr_from_mse(blind_mse_in))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(
                f"Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | "
                f"MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)"
            )


if __name__ == '__main__':
    main()
