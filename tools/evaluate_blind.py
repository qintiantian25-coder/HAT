#!/usr/bin/env python3
"""Evaluate full-image and blind-pixel metrics for model outputs.

Usage:
  python tools/evaluate_blind.py --out_dir /path/to/outputs --gt_dir /path/to/gts \
    --input_dir /path/to/test_blur --mask_root /path/to/test_mask --save_dir /path/to/save
"""

import argparse
import csv
import os
import re

import cv2
import numpy as np

from hat.utils.metric_utils import psnr_uint8, ssim_uint8


def natural_sort_key(text):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r'([0-9]+)', text)]


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

    if not coords:
        return None
    return np.unique(np.array(coords, dtype=np.int32), axis=0)


def find_mask_csv(mask_root, rel_path):
    if not mask_root:
        return None

    rel_dir = os.path.dirname(rel_path)
    base_dir = os.path.basename(rel_dir)
    candidates = [
        os.path.join(mask_root, rel_dir, 'blind_coords.csv'),
        os.path.join(mask_root, rel_dir, 'blind_pixel_coords.csv'),
        os.path.join(mask_root, base_dir, 'blind_coords.csv'),
        os.path.join(mask_root, base_dir, 'blind_pixel_coords.csv'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate(out_dir, gt_dir, input_dir=None, mask_csv=None, mask_root=None, save_dir=None):
    out_files = []
    for root, _, files in os.walk(out_dir):
        for file_name in files:
            if file_name.lower().endswith('.png'):
                out_files.append(os.path.join(root, file_name))
    out_files.sort(key=lambda path: natural_sort_key(os.path.relpath(path, out_dir)))

    gt_map = {}
    for root, _, files in os.walk(gt_dir):
        for file_name in files:
            if file_name.lower().endswith('.png'):
                abs_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(abs_path, gt_dir)
                gt_map[rel_path] = abs_path
                gt_map[os.path.basename(rel_path)] = abs_path

    input_map = {}
    if input_dir and os.path.exists(input_dir):
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                if file_name.lower().endswith('.png'):
                    abs_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(abs_path, input_dir)
                    input_map[rel_path] = abs_path
                    input_map[os.path.basename(rel_path)] = abs_path

    global_blind_coords = load_blind_coords(mask_csv)

    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []

    print(f'===> Evaluating {len(out_files)} output images...')
    for out_path in out_files:
        rel_path = os.path.relpath(out_path, out_dir)
        gt_path = gt_map.get(rel_path) or gt_map.get(os.path.basename(rel_path))
        if not gt_path or not os.path.exists(out_path):
            continue

        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if out_img is None or gt_img is None:
            continue
        if out_img.shape != gt_img.shape:
            out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

        row = {
            'image': rel_path,
            'psnr': float(psnr_uint8(gt_img, out_img)),
            'ssim': ssim_uint8(gt_img, out_img),
            'blind_mae': None,
            'blind_rmse': None,
            'blind_psnr': None,
            'blind_mae_input': None,
            'blind_mae_gain_abs': None,
            'blind_mae_gain_pct': None,
            'blind_count': 0,
        }

        blind_coords = global_blind_coords
        if blind_coords is None and mask_root:
            blind_coords = load_blind_coords(find_mask_csv(mask_root, rel_path))

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

                in_path = input_map.get(rel_path) or input_map.get(os.path.basename(rel_path))
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
                    'blind_count': int(len(err)),
                })
                if in_mae is not None:
                    row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                    row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

        per_image_logs.append(row)

    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))
        print('===> Blind-Pixel Focused Metrics')
        if mask_csv:
            print(f'BlindCoordsCSV: {mask_csv}')
        elif mask_root:
            print(f'BlindCoordsRoot: {mask_root}')
        print(f'BlindCount(total sampled): {blind_pix_sum}')
        print(f'Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}')

        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse_in, 1e-12)))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(
                f'Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | '
                f'MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)'
            )

    if save_dir is None:
        save_dir = os.path.join(out_dir, '..', 'eval')
    os.makedirs(save_dir, exist_ok=True)
    save_csv = os.path.join(save_dir, 'test_blind_metrics.csv')
    if per_image_logs:
        keys = [
            'image', 'psnr', 'ssim',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count',
        ]
        with open(save_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_logs:
                writer.writerow(row)
        print(f'Per-image test metrics saved to: {save_csv}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='Directory with model output PNGs')
    parser.add_argument('--gt_dir', required=True, help='Ground-truth images directory')
    parser.add_argument('--input_dir', default=None, help='Input LR images dir (for input blind error)')
    parser.add_argument('--mask_csv', default=None, help='Single CSV with blind pixel coords')
    parser.add_argument('--mask_root', default=None, help='Root directory with per-group blind CSV files')
    parser.add_argument('--save_dir', default=None, help='Directory to save per-image metrics CSV')
    args = parser.parse_args()

    evaluate(args.out_dir, args.gt_dir, args.input_dir, args.mask_csv, args.mask_root, args.save_dir)


if __name__ == '__main__':
    main()
