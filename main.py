#!/usr/bin/env python3
"""Unified entrypoint for blind restoration training/testing.

Usage:
  python main.py --train --config_path ./experiment.cfg
  python main.py --test --config_path ./experiment.cfg
"""
from __future__ import annotations

import argparse
import configparser
import os
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def as_int(value: str, default: int) -> int:
    if value is None or str(value).strip() == '':
        return default
    return int(value)


def as_float(value: str, default: float) -> float:
    if value is None or str(value).strip() == '':
        return default
    return float(value)


def as_list(value: str, default=None, sep=','):
    if value is None or str(value).strip() == '':
        return default if default is not None else []
    return [item.strip() for item in value.split(sep) if item.strip()]


def count_pngs(root_dir):
    if not root_dir or not os.path.exists(root_dir):
        return 0
    total = 0
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith('.png'):
                total += 1
    return total


def yaml_scalar(value):
    if value is None:
        return '~'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value)
    if text == '':
        return "''"
    return "'" + text.replace("'", "''") + "'"


def dump_yaml(obj, indent=0):
    lines = []
    pad = ' ' * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f'{pad}{key}:')
                lines.extend(dump_yaml(value, indent + 2))
            elif isinstance(value, list):
                if len(value) == 0:
                    lines.append(f'{pad}{key}: []')
                else:
                    lines.append(f'{pad}{key}:')
                    for item in value:
                        if isinstance(item, (dict, list)):
                            lines.append(f'{pad}  -')
                            lines.extend(dump_yaml(item, indent + 4))
                        else:
                            lines.append(f'{pad}  - {yaml_scalar(item)}')
            else:
                lines.append(f'{pad}{key}: {yaml_scalar(value)}')
    else:
        raise TypeError(f'Unsupported YAML object type: {type(obj)!r}')
    return lines


def parse_cfg(cfg_path):
    parser = configparser.ConfigParser(interpolation=None)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        parser.read_file(f)
    return parser


def get_section(cfg, section):
    return cfg[section] if cfg.has_section(section) else {}


def build_common(cfg):
    common = get_section(cfg, 'common')
    paths = get_section(cfg, 'paths')
    train_sec = get_section(cfg, 'train')
    test_sec = get_section(cfg, 'test')
    val_sec = get_section(cfg, 'val')
    logger_sec = get_section(cfg, 'logger')

    dataset_root = common.get('dataset_root', '/home/student_server/Qtt/NAFNet/data_new')
    exp_root = common.get('experiments_root', './experiments')
    name = common.get('name', 'train_HAT_blind_restoration')
    model_path = common.get('model_path', str(Path(exp_root) / 'models' / 'best_model.pt'))

    return {
        'name': name,
        'dataset_root': dataset_root,
        'experiments_root': exp_root,
        'model_path': model_path,
        'train_blur': paths.get('train_blur', paths.get('train_lq', os.path.join(dataset_root, 'train_blur'))),
        'train_sharp': paths.get('train_sharp', paths.get('train_gt', os.path.join(dataset_root, 'train_sharp'))),
        'val_blur': paths.get('val_blur', paths.get('val_lq', os.path.join(dataset_root, 'val_blur'))),
        'val_sharp': paths.get('val_sharp', paths.get('val_gt', os.path.join(dataset_root, 'val_sharp'))),
        'val_mask_root': val_sec.get('val_mask_root', paths.get('val_mask', os.path.join(dataset_root, 'val_mask'))),
        'test_blur': paths.get('test_blur', os.path.join(dataset_root, 'test_blur')),
        'test_sharp': paths.get('test_sharp', os.path.join(dataset_root, 'test_sharp')),
        'test_mask_root': paths.get('test_mask', os.path.join(dataset_root, 'test_mask')),
        'num_gpu': as_int(common.get('num_gpu'), 1),
        'scale': as_int(common.get('scale'), 1),
        'in_chans': as_int(common.get('in_chans'), 1),
        'img_size': as_int(common.get('img_size'), 64),
        'window_size': as_int(common.get('window_size'), 16),
        'use_checkpoint': as_bool(common.get('use_checkpoint'), False),
        'upscaler': common.get('upscaler', ''),
        'save_blind_eval_dir': test_sec.get('blind_eval_dir', str(Path(exp_root) / 'blind_eval')),
        'test_dataset_name': test_sec.get('test_dataset_name', 'BlindTest'),
        'save_img': as_bool(test_sec.get('save_img'), True),
        'pbar': as_bool(test_sec.get('pbar'), True),
        'suffix': test_sec.get('suffix', ''),
        'gt_size': as_int(train_sec.get('gt_size'), 128),
        'batch_size_per_gpu': as_int(train_sec.get('batch_size_per_gpu'), 2),
        'num_worker_per_gpu': as_int(train_sec.get('num_worker_per_gpu'), 2),
        'gradient_accumulation_steps': as_int(train_sec.get('gradient_accumulation_steps'), 2),
        'use_amp': as_bool(train_sec.get('use_amp'), True),
        'ema_decay': as_float(train_sec.get('ema_decay'), 0.999),
        'total_iter': as_int(train_sec.get('total_iter'), 150000),
        'warmup_iter': as_int(train_sec.get('warmup_iter'), 1500),
        'lr': as_float(train_sec.get('lr'), 1e-4),
        'weight_decay': as_float(train_sec.get('weight_decay'), 1e-4),
        'milestones': as_list(train_sec.get('milestones'), ['75000', '120000', '135000', '145000']),
        'gamma': as_float(train_sec.get('gamma'), 0.5),
        'val_freq_epochs': as_int(train_sec.get('val_freq_epochs'), 20),
        'save_checkpoint_freq': as_int(logger_sec.get('save_checkpoint_freq'), as_int(train_sec.get('save_checkpoint_freq'), 0)),
        'pretrain_network_g': common.get('pretrain_network_g', ''),
    }


def estimate_val_freq_iters(c):
    train_image_count = count_pngs(c['train_blur'])
    if train_image_count <= 0:
        return 5000  # fallback when image count unknown

    effective_batch = max(1, c['batch_size_per_gpu'] * c['num_gpu'])
    iters_per_epoch = max(1, math.ceil(train_image_count / effective_batch))
    return max(1, iters_per_epoch * max(1, c['val_freq_epochs']))


def build_train_options(c):
    val_freq_iters = estimate_val_freq_iters(c)

    train_dataset = {
        'name': 'BlindDataset',
        'type': 'BlindPairedImageDataset',
        'dataroot_gt': c['train_sharp'],
        'dataroot_lq': c['train_blur'],
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'gt_size': c['gt_size'],
        'use_hflip': True,
        'use_rot': True,
        'num_worker_per_gpu': c['num_worker_per_gpu'],
        'batch_size_per_gpu': c['batch_size_per_gpu'],
        'dataset_enlarge_ratio': 1,
        'prefetch_mode': None,
        'pin_memory': True,
        'scale': c['scale'],
    }

    val_dataset = {
        'name': 'BlindVal',
        'type': 'BlindPairedImageDataset',
        'dataroot_gt': c['val_sharp'],
        'dataroot_lq': c['val_blur'],
        'io_backend': {'type': 'disk'},
        'phase': 'val',
        'scale': c['scale'],
    }

    return {
        'name': c['name'],
        'model_type': 'HATModel',
        'scale': c['scale'],
        'num_gpu': c['num_gpu'],
        'manual_seed': 0,
        'datasets': {
            'train': train_dataset,
            'val_1': val_dataset,
        },
        'network_g': {
            'type': 'HAT',
            'upscale': c['scale'],
            'in_chans': c['in_chans'],
            'img_size': c['img_size'],
            'window_size': c['window_size'],
            'compress_ratio': 3,
            'squeeze_factor': 30,
            'conv_scale': 0.01,
            'overlap_ratio': 0.5,
            'img_range': 1.0,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': c['upscaler'],
            'resi_connection': '1conv',
            'use_checkpoint': c['use_checkpoint'],
        },
        'path': {
            'pretrain_network_g': c['pretrain_network_g'] or None,
            'param_key_g': 'params_ema',
            'strict_load_g': False,
            'resume_state': None,
            'experiments_root': c['experiments_root'],
            'models': str(Path(c['experiments_root']) / 'models'),
            'visualization': str(Path(c['experiments_root']) / 'visualization'),
        },
        'train': {
            'gradient_accumulation_steps': c['gradient_accumulation_steps'],
            'use_amp': c['use_amp'],
            'ema_decay': c['ema_decay'],
            'optim_g': {
                'type': 'Adam',
                'lr': c['lr'],
                'weight_decay': c['weight_decay'],
                'betas': [0.9, 0.999],
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [int(v) for v in c['milestones']],
                'gamma': c['gamma'],
            },
            'total_iter': c['total_iter'],
            'warmup_iter': c['warmup_iter'],
            'pixel_opt': {
                'type': 'L1Loss',
                'loss_weight': 1.0,
                'reduction': 'mean',
            },
        },
        'val': {
            'val_freq': val_freq_iters,
            'save_img': True,
            'pbar': True,
            'mask_root': c['val_mask_root'],
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False, 'better': 'higher'},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False, 'better': 'higher'},
            },
        },
        'logger': {
            'print_freq': 100,
            'save_checkpoint_freq': int(c['save_checkpoint_freq']) if int(c['save_checkpoint_freq']) > 0 else 5000,
            'use_tb_logger': True,
            'wandb': {'project': None, 'resume_id': None},
            'training_log_file': str(Path(c['experiments_root']) / 'logs' / 'training.txt'),
            'validation_log_file': str(Path(c['experiments_root']) / 'logs' / 'validation.txt'),
        },
    }


def build_test_options(c):
    return {
        'name': c['name'],
        'model_type': 'HATModel',
        'scale': c['scale'],
        'num_gpu': c['num_gpu'],
        'manual_seed': 0,
        'datasets': {
            'test_1': {
                'name': c['test_dataset_name'],
                'type': 'BlindPairedImageDataset',
                'dataroot_gt': c['test_sharp'],
                'dataroot_lq': c['test_blur'],
                'io_backend': {'type': 'disk'},
                'phase': 'test',
                'scale': c['scale'],
            },
        },
        'network_g': {
            'type': 'HAT',
            'upscale': c['scale'],
            'in_chans': c['in_chans'],
            'img_size': c['img_size'],
            'window_size': c['window_size'],
            'compress_ratio': 3,
            'squeeze_factor': 30,
            'conv_scale': 0.01,
            'overlap_ratio': 0.5,
            'img_range': 1.0,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': c['upscaler'],
            'resi_connection': '1conv',
            'use_checkpoint': False,
        },
        'path': {
            'pretrain_network_g': c['model_path'],
            'strict_load_g': False,
            'param_key_g': None,
            'visualization': str(Path(c['experiments_root']) / 'visualization'),
        },
        'val': {
            'suffix': c['suffix'],
            'save_img': c['save_img'],
            'pbar': c['pbar'],
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False},
            },
        },
        'logger': {
            'print_freq': 100,
            'use_tb_logger': False,
            'validation_log_file': str(Path(c['experiments_root']) / 'logs' / 'validation.txt'),
        },
    }


def write_temp_yaml(opt_dict, prefix):
    yaml_text = '\n'.join(dump_yaml(opt_dict)) + '\n'
    tmp = tempfile.NamedTemporaryFile(mode='w', prefix=prefix, suffix='.yml', delete=False, encoding='utf-8')
    tmp.write(yaml_text)
    tmp.flush()
    tmp.close()
    return tmp.name


def ensure_log_dir(c, opt=None):
    log_dirs = {Path(c['experiments_root']) / 'logs'}

    if isinstance(opt, dict):
        logger_cfg = opt.get('logger', {})
        if isinstance(logger_cfg, dict):
            for key in ('training_log_file', 'validation_log_file'):
                log_file = logger_cfg.get(key)
                if log_file:
                    log_dirs.add(Path(log_file).parent)

    for log_dir in log_dirs:
        log_dir.mkdir(parents=True, exist_ok=True)

    return Path(c['experiments_root']) / 'logs'


def resolve_training_log_file(c, opt=None):
    default_log = Path(c['experiments_root']) / 'logs' / 'training.txt'
    if not isinstance(opt, dict):
        return default_log
    logger_cfg = opt.get('logger', {})
    if isinstance(logger_cfg, dict):
        path = logger_cfg.get('training_log_file')
        if path:
            return Path(path)
    return default_log


def tee_subprocess(command, log_file, cwd=ROOT):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write(' '.join(command) + '\n')
        f.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def run_command(command, cwd=ROOT):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    print(' '.join(command))
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()

    cfg = parse_cfg(args.config_path)
    common = build_common(cfg)

    if args.train:
        opt = build_train_options(common)
        ensure_log_dir(common, opt)
        temp_yaml = write_temp_yaml(opt, 'hat_train_')
        print(f'Training config written to: {temp_yaml}')
        print(f'Best model will be saved under: {Path(common["experiments_root"]) / "models"}')
        tee_subprocess(
            [sys.executable, 'hat/train.py', '-opt', temp_yaml],
            resolve_training_log_file(common, opt)
        )
        return

    if args.test:
        # auto-fallback to latest model if best_model.pt not found
        model_path = Path(common['model_path'])
        if not model_path.exists():
            latest_model = Path(common['experiments_root']) / 'models' / 'net_g_latest.pth'
            if latest_model.exists():
                print(f"WARNING: {model_path} not found, falling back to latest model: {latest_model}")
                model_path = latest_model
                common['model_path'] = str(latest_model)
            else:
                raise FileNotFoundError(
                    f'Best model not found: {model_path}, and no latest model found. '
                    'Run training first.'
                )

        opt = build_test_options(common)
        ensure_log_dir(common, opt)
        temp_yaml = write_temp_yaml(opt, 'hat_test_')
        print(f'Test config written to: {temp_yaml}')
        run_command([sys.executable, 'hat/test.py', '-opt', temp_yaml])

        out_dir = Path(common['experiments_root']) / 'visualization' / common['test_dataset_name']
        eval_cmd = [
            sys.executable, 'tools/evaluate_blind.py',
            '--out_dir', str(out_dir),
            '--gt_dir', common['test_sharp'],
            '--input_dir', common['test_blur'],
            '--mask_root', common['test_mask_root'],
            '--save_dir', common['save_blind_eval_dir'],
            '--save_triple',
        ]
        run_command(eval_cmd)


if __name__ == '__main__':
    main()
