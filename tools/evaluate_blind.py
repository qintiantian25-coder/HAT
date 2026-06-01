#!/usr/bin/env python3
"""Evaluate full-image and blind-pixel metrics for model outputs.

Usage:
  python tools/evaluate_blind.py --out_dir /path/to/outputs --gt_dir /path/to/gts \
    --input_dir /path/to/test_blur --mask_root /path/to/test_mask --save_dir /path/to/save \
    [--save_triple]
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
    """Load static blind pixel coordinates from CSV (columns: x, y)."""
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


def load_flash_map(csv_path):
    """Load per-frame flash pixel coordinates from CSV (columns: frame_name, x, y, ...).

    Returns:
        dict mapping frame basename -> list of (x, y) tuples, or {} if unavailable.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'frame_name' not in reader.fieldnames:
            return {}
        if 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                fname = os.path.basename(row['frame_name'])
                x = int(float(row['x']))
                y = int(float(row['y']))
            except Exception:
                continue
            flash_map.setdefault(fname, set()).add((x, y))

    # Deduplicate and convert sets to sorted lists
    for k in list(flash_map.keys()):
        flash_map[k] = list(flash_map[k])
    return flash_map


def find_mask_csv(mask_root, rel_path):
    """Locate blind_pixel_coords.csv or blind_coords.csv for a given output image."""
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


def find_flash_csv(mask_root, rel_path):
    """Locate flash_pixel_coords.csv for a given output image."""
    if not mask_root:
        return None

    rel_dir = os.path.dirname(rel_path)
    base_dir = os.path.basename(rel_dir)
    candidates = [
        os.path.join(mask_root, rel_dir, 'flash_pixel_coords.csv'),
        os.path.join(mask_root, base_dir, 'flash_pixel_coords.csv'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate(out_dir, gt_dir, input_dir=None, mask_csv=None, mask_root=None,
             save_dir=None, save_triple=False, write_split_csv=True, write_summary_csv=True):
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

    # Auto-enable triple comparison when input_dir is available
    if save_triple is None:
        save_triple = bool(input_dir and os.path.exists(input_dir))
    if save_triple and save_dir:
        triple_root = os.path.join(save_dir, 'triple_comparison')

    # Global blind coords (single CSV mode)
    global_blind_coords = load_blind_coords(mask_csv)

    # Cache flash maps per group to avoid re-reading the same CSV
    flash_cache = {}  # group_name -> flash_map dict

    # Per-image and per-seq accumulators
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []
    seq_stats = {}  # seq_name -> {blind_abs_sum, blind_sq_sum, blind_abs_in_sum, blind_sq_in_sum, blind_pix_sum}

    print(f'===> Evaluating {len(out_files)} output images...')
    for out_path in out_files:
        rel_path = os.path.relpath(out_path, out_dir)
        img_name = os.path.basename(rel_path)
        seq_name = rel_path.split(os.sep)[0] if os.sep in rel_path else ''

        gt_path = gt_map.get(rel_path) or gt_map.get(os.path.basename(rel_path))
        if not gt_path or not os.path.exists(out_path):
            continue

        out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if out_img is None or gt_img is None:
            continue
        if out_img.shape != gt_img.shape:
            out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

        # --- Triple comparison ---
        if save_triple and save_dir and input_dir:
            in_path = input_map.get(rel_path) or input_map.get(img_name)
            if in_path and os.path.exists(in_path):
                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is not None:
                    if in_img.shape != gt_img.shape:
                        in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                    # Create separator bar (2px white line)
                    sep = np.full((gt_img.shape[0], 2), 255, dtype=np.uint8)
                    triple = np.hstack([in_img, sep, out_img, sep, gt_img])
                    triple_rel_dir = os.path.dirname(rel_path)
                    triple_dir = os.path.join(triple_root, triple_rel_dir) if triple_rel_dir else triple_root
                    os.makedirs(triple_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(triple_dir, f'triple_{img_name}'), triple)

        # --- Full-image metrics ---
        row = {
            'image': rel_path,
            'seq': seq_name,
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

        # --- Blind + Flash pixel metrics ---
        # 1. Load static blind coords
        blind_coords = global_blind_coords
        if blind_coords is None and mask_root:
            blind_coords = load_blind_coords(find_mask_csv(mask_root, rel_path))

        # 2. Load per-frame flash coords
        flash_coords_list = []
        if mask_root:
            # Determine group for flash CSV caching
            seq_hint = seq_name or ''
            cache_key = seq_hint
            if cache_key not in flash_cache:
                flash_csv_path = find_flash_csv(mask_root, rel_path)
                flash_cache[cache_key] = load_flash_map(flash_csv_path)
            flash_map = flash_cache[cache_key]
            frame_flash = flash_map.get(img_name, [])
            if frame_flash:
                flash_coords_list.extend(frame_flash)

        # 3. Merge blind + flash coords
        all_coords = []
        h, w = gt_img.shape[:2]
        if blind_coords is not None:
            bx = blind_coords[:, 0]
            by = blind_coords[:, 1]
            valid = (bx >= 0) & (bx < w) & (by >= 0) & (by < h)
            if np.any(valid):
                all_coords.extend(zip(bx[valid].tolist(), by[valid].tolist()))

        for (fx, fy) in flash_coords_list:
            if 0 <= fx < w and 0 <= fy < h:
                all_coords.append((fx, fy))

        if all_coords:
            coords_arr = np.unique(np.array(all_coords, dtype=np.int32), axis=0)
            if coords_arr.size > 0:
                x = coords_arr[:, 0]
                y = coords_arr[:, 1]
                gt_vals = gt_img[y, x].astype(np.float64)
                out_vals = out_img[y, x].astype(np.float64)
                err = out_vals - gt_vals

                blind_abs = np.abs(err)
                blind_sq = err ** 2

                blind_abs_sum_local = float(blind_abs.sum())
                blind_sq_sum_local = float(blind_sq.sum())
                blind_count = int(len(err))

                blind_abs_sum += blind_abs_sum_local
                blind_sq_sum += blind_sq_sum_local
                blind_pix_sum += blind_count

                # Per-seq accumulators
                if seq_name not in seq_stats:
                    seq_stats[seq_name] = {
                        'blind_abs_sum': 0.0, 'blind_sq_sum': 0.0,
                        'blind_abs_in_sum': 0.0, 'blind_sq_in_sum': 0.0,
                        'blind_pix_sum': 0, 'image_count': 0,
                    }
                st = seq_stats[seq_name]
                st['blind_abs_sum'] += blind_abs_sum_local
                st['blind_sq_sum'] += blind_sq_sum_local
                st['blind_pix_sum'] += blind_count

                # Input blind error
                in_path = input_map.get(rel_path) or input_map.get(img_name)
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
                        in_abs_sum = float(in_abs.sum())
                        in_sq_sum = float(in_sq.sum())
                        blind_abs_in_sum += in_abs_sum
                        blind_sq_in_sum += in_sq_sum
                        st['blind_abs_in_sum'] += in_abs_sum
                        st['blind_sq_in_sum'] += in_sq_sum
                        in_mae = float(in_abs.mean())

                row.update({
                    'blind_mae': float(blind_abs.mean()),
                    'blind_rmse': float(np.sqrt(blind_sq.mean())),
                    'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                    'blind_mae_input': in_mae,
                    'blind_count': blind_count,
                })
                if in_mae is not None:
                    row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                    row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

        # Track per-seq image count (even for images without blind pixels)
        if seq_name not in seq_stats:
            seq_stats[seq_name] = {
                'blind_abs_sum': 0.0, 'blind_sq_sum': 0.0,
                'blind_abs_in_sum': 0.0, 'blind_sq_in_sum': 0.0,
                'blind_pix_sum': 0, 'image_count': 0,
            }
        seq_stats[seq_name]['image_count'] += 1

        per_image_logs.append(row)

    # --- Save per-image CSV ---
    if save_dir is None:
        save_dir = os.path.join(out_dir, '..', 'eval')
    os.makedirs(save_dir, exist_ok=True)
    save_csv = os.path.join(save_dir, 'test_blind_metrics.csv')
    if per_image_logs:
        keys = [
            'image', 'seq', 'psnr', 'ssim',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count',
        ]
        with open(save_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_logs:
                writer.writerow(row)
        print(f'Per-image test metrics saved to: {save_csv}')

    # --- Save per-seq summary CSV ---
    summary_keys = [
        'seq', 'images', 'blind_count',
        'blind_mae', 'blind_rmse', 'blind_psnr',
        'input_blind_mae', 'input_blind_psnr',
        'blind_mae_gain_abs', 'blind_mae_gain_pct',
    ]
    summary_rows = []
    for seq_name in sorted(seq_stats.keys(), key=natural_sort_key):
        st = seq_stats[seq_name]
        pix = int(st['blind_pix_sum'])
        row = {
            'seq': seq_name if seq_name else 'root',
            'images': st['image_count'],
            'blind_count': pix,
            'blind_mae': None,
            'blind_rmse': None,
            'blind_psnr': None,
            'input_blind_mae': None,
            'input_blind_psnr': None,
            'blind_mae_gain_abs': None,
            'blind_mae_gain_pct': None,
        }
        if pix > 0:
            mae = st['blind_abs_sum'] / pix
            mse = st['blind_sq_sum'] / pix
            row['blind_mae'] = float(mae)
            row['blind_rmse'] = float(np.sqrt(mse))
            row['blind_psnr'] = float(10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12)))
            if st['blind_abs_in_sum'] > 0:
                in_mae = st['blind_abs_in_sum'] / pix
                in_mse = st['blind_sq_in_sum'] / pix
                row['input_blind_mae'] = float(in_mae)
                row['input_blind_psnr'] = float(10.0 * np.log10((255.0 * 255.0) / max(in_mse, 1e-12)))
                row['blind_mae_gain_abs'] = float(in_mae - mae)
                row['blind_mae_gain_pct'] = float(100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12))
        summary_rows.append(row)

    if summary_rows:
        summary_csv = os.path.join(save_dir, 'test_blind_summary_by_seq.csv')
        with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f'Per-seq summary saved to: {summary_csv}')

    # --- Save per-seq split CSVs (one per group) ---
    if write_split_csv and per_image_logs:
        seq_groups = {}
        for row in per_image_logs:
            seq = row.get('seq', '') or 'root'
            seq_groups.setdefault(seq, []).append(row)

        for seq_name in sorted(seq_groups.keys(), key=natural_sort_key):
            seq_rows = seq_groups[seq_name]
            if not seq_rows:
                continue
            seq_label = seq_name if seq_name != 'root' else 'root'
            seq_csv = os.path.join(save_dir, f'test_blind_metrics_{seq_label}.csv')
            with open(seq_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in seq_rows:
                    writer.writerow(row)
            print(f'Per-seq metrics saved to: {seq_csv}')

    # --- Aggregate PSNR/SSIM summary ---
    psnr_vals = [r['psnr'] for r in per_image_logs if r.get('psnr') is not None]
    ssim_vals = [r['ssim'] for r in per_image_logs if r.get('ssim') is not None]
    if psnr_vals or ssim_vals:
        print('===> Aggregate Full-Image Metrics')
        if psnr_vals:
            avg_psnr = sum(psnr_vals) / len(psnr_vals)
            print(f'Average PSNR: {avg_psnr:.4f} ({len(psnr_vals)} images)')
        if ssim_vals:
            avg_ssim = sum(ssim_vals) / len(ssim_vals)
            print(f'Average SSIM: {avg_ssim:.4f} ({len(ssim_vals)} images)')

    # --- Console summary ---
    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))
        print('===> Blind-Pixel Focused Metrics')
        if mask_csv:
            print(f'BlindCoordsCSV: {mask_csv}')
        elif mask_root:
            print(f'MaskRoot: {mask_root}')
        print(f'BlindCount (total sampled, incl. flash): {blind_pix_sum}')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help='Directory with model output PNGs')
    parser.add_argument('--gt_dir', required=True, help='Ground-truth images directory')
    parser.add_argument('--input_dir', default=None, help='Input LR images dir (for input blind error)')
    parser.add_argument('--mask_csv', default=None, help='Single CSV with blind pixel coords')
    parser.add_argument('--mask_root', default=None, help='Root directory with per-group blind/flash CSV files')
    parser.add_argument('--save_dir', default=None, help='Directory to save per-image metrics CSV')
    parser.add_argument('--save_triple', action='store_true', default=None,
                        help='Save triple comparison [input|output|GT] images')
    parser.add_argument('--no_split_csv', action='store_true', default=False,
                        help='Disable per-seq split CSV files')
    parser.add_argument('--no_summary_csv', action='store_true', default=False,
                        help='Disable per-seq summary CSV file')
    args = parser.parse_args()

    evaluate(args.out_dir, args.gt_dir, args.input_dir, args.mask_csv, args.mask_root,
             args.save_dir, args.save_triple,
             write_split_csv=not args.no_split_csv,
             write_summary_csv=not args.no_summary_csv)


if __name__ == '__main__':
    main()
