import csv
import os
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from hat.utils.metric_utils import ensure_gray_uint8, psnr_uint8, ssim_uint8


def _resolve_validation_log_file(opt):
    logger_cfg = opt.get('logger', {}) if isinstance(opt, dict) else {}
    if isinstance(logger_cfg, dict):
        log_file = logger_cfg.get('validation_log_file')
        if log_file:
            return log_file

    path_cfg = opt.get('path', {}) if isinstance(opt, dict) else {}
    if isinstance(path_cfg, dict):
        old_value = path_cfg.get('validation_log')
        if not old_value:
            return None
        if str(old_value).lower().endswith('.txt'):
            return old_value
        return os.path.join(old_value, 'validation.txt')

    return None


def _load_blind_coords(csv_path):
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


def _derive_mask_csv(mask_root, lq_path, lq_root):
    """Derive blind_coords.csv path from lq_path by mirroring directory structure."""
    rel = os.path.relpath(lq_path, lq_root)
    group_id = rel.split(os.sep)[0]
    candidates = [
        os.path.join(mask_root, group_id, 'blind_coords.csv'),
        os.path.join(mask_root, group_id, 'blind_pixel_coords.csv'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@MODEL_REGISTRY.register()
class HATModel(SRModel):

    def save(self, epoch, current_iter):
        return

    def save_network(self, net, net_label, current_iter, param_key='params'):
        return

    def save_training_state(self, epoch, current_iter):
        return

    def pre_process(self):
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # blind pixel metric accumulators
        blind_abs_sum = 0.0
        blind_sq_sum = 0.0
        blind_pix_sum = 0

        # discover mask root from val config (not dataset - model is pure HAT, mask only for evaluation)
        mask_root = self.opt['val'].get('mask_root', None)
        lq_root = dataloader.dataset.opt.get('dataroot_lq', None)
        if mask_root and not os.path.exists(mask_root):
            mask_root = None

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            lq_path = val_data['lq_path'][0]
            dataset_root = lq_root or dataloader.dataset.opt.get('dataroot_lq')
            if dataset_root:
                img_name = os.path.splitext(os.path.relpath(lq_path, dataset_root))[0]
            else:
                img_name = os.path.splitext(os.path.basename(lq_path))[0]

            self.feed_data(val_data)
            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

                sr_gray = ensure_gray_uint8(sr_img)
                gt_gray = ensure_gray_uint8(gt_img)

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_type = str(opt_.get('type', '')).lower()
                    crop_border = int(opt_.get('crop_border', 0))
                    if metric_type == 'calculate_psnr':
                        self.metric_results[name] += psnr_uint8(gt_gray, sr_gray, crop_border)
                    elif metric_type == 'calculate_ssim':
                        ssim_val = ssim_uint8(gt_gray, sr_gray, crop_border)
                        self.metric_results[name] += 0.0 if ssim_val is None else ssim_val
                    else:
                        raise NotImplementedError(f'Unsupported validation metric: {metric_type}')

            # blind pixel metrics
            if mask_root and lq_root:
                csv_path = _derive_mask_csv(mask_root, lq_path, lq_root)
                blind_coords = _load_blind_coords(csv_path)
                if blind_coords is not None:
                    h, w = gt_gray.shape[:2]
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]
                        gt_vals = gt_gray[y, x].astype(np.float64)
                        out_vals = sr_gray[y, x].astype(np.float64)
                        err = out_vals - gt_vals
                        blind_abs_sum += float(np.abs(err).sum())
                        blind_sq_sum += float((err ** 2).sum())
                        blind_pix_sum += int(len(err))

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # log blind pixel metrics
        if blind_pix_sum > 0:
            blind_mae = blind_abs_sum / blind_pix_sum
            blind_mse = blind_sq_sum / blind_pix_sum
            blind_rmse = float(np.sqrt(blind_mse))
            blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))

            logger = get_root_logger()
            logger.info(
                f'Validation iter={current_iter} dataset={dataset_name} '
                f'Blind MAE={blind_mae:.6f} Blind RMSE={blind_rmse:.6f} Blind PSNR={blind_psnr:.3f} '
                f'(sampled {blind_pix_sum} blind pixels)'
            )

            validation_log = _resolve_validation_log_file(self.opt)
            if validation_log:
                try:
                    log_dir = Path(validation_log).parent
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with open(validation_log, 'a', encoding='utf-8') as f:
                        f.write(
                            f'iter={current_iter} dataset={dataset_name} '
                            f'Blind_MAE={blind_mae:.6f} Blind_RMSE={blind_rmse:.6f} '
                            f'Blind_PSNR={blind_psnr:.3f} Blind_Count={blind_pix_sum}\n'
                        )
                except Exception:
                    pass

    def _update_best_metric_result(self, dataset_name, metric, value, current_iter):
        try:
            super(HATModel, self)._update_best_metric_result(dataset_name, metric, value, current_iter)
        except Exception:
            pass

        if metric is None:
            return
        if str(metric).lower() != 'psnr':
            return

        root_path = self.opt.get('root_path') or os.getcwd()
        models_dir = os.path.join(root_path, 'experiments', 'models')
        try:
            os.makedirs(models_dir, exist_ok=True)
        except Exception:
            pass

        meta_file = os.path.join(models_dir, 'best_model_meta.json')
        prev_best = -1e9
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    prev_best = float(j.get('best_psnr', prev_best))
            except Exception:
                prev_best = -1e9

        if value is None:
            return
        try:
            cur = float(value)
        except Exception:
            return

        if cur > prev_best:
            net = getattr(self, 'net_g_ema', None) or getattr(self, 'net_g', None)
            if net is None:
                return
            save_path = os.path.join(models_dir, 'best_model.pt')
            try:
                torch.save(net.state_dict(), save_path)
                meta = {'best_psnr': cur, 'iter': int(current_iter), 'time': time.time()}
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta, f)
                logger = get_root_logger()
                logger.info(f'Best model updated: {save_path} (psnr={cur:.6f})')
                validation_log = _resolve_validation_log_file(self.opt)
                if validation_log:
                    try:
                        with open(validation_log, 'a', encoding='utf-8') as f:
                            f.write(f'iter={current_iter} best_model_updated psnr={cur:.6f} path={save_path}\n')
                    except Exception:
                        pass
            except Exception as e:
                logger = get_root_logger()
                logger.error(f'Failed to save best model: {e}')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        try:
            super(HATModel, self)._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        except Exception:
            pass

        try:
            logger = get_root_logger()
            metric_parts = []
            for key, value in getattr(self, 'metric_results', {}).items():
                try:
                    metric_parts.append(f'{key}={float(value):.6f}')
                except Exception:
                    metric_parts.append(f'{key}={value}')
            if metric_parts:
                logger.info(f'Validation iter={current_iter} dataset={dataset_name} ' + ' '.join(metric_parts))
        except Exception:
            pass

        validation_log = _resolve_validation_log_file(self.opt)
        if not validation_log:
            return

        try:
            log_dir = Path(validation_log).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            metric_parts = []
            for key, value in getattr(self, 'metric_results', {}).items():
                try:
                    metric_parts.append(f'{key}={float(value):.6f}')
                except Exception:
                    metric_parts.append(f'{key}={value}')
            line = f'iter={current_iter} dataset={dataset_name} ' + ' '.join(metric_parts) + '\n'
            with open(validation_log, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass

    def optimize_parameters(self, current_iter):
        if not hasattr(self, 'optimizer_g'):
            raise RuntimeError('optimizer_g not found on model')

        try:
            if not hasattr(self, 'net_g'):
                raise RuntimeError('net_g not found on model')
            if not hasattr(self, 'lq'):
                raise RuntimeError('lq (input) not available; feed_data not called')

            self.net_g.train()
            self.output = self.net_g(self.lq)

            l_g_total = 0
            loss_dict = OrderedDict()
            if hasattr(self, 'cri_pix') and self.cri_pix is not None and hasattr(self, 'gt'):
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total = l_g_total + l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if isinstance(l_g_total, torch.Tensor):
                if l_g_total.requires_grad:
                    self.optimizer_g.zero_grad()
                    l_g_total.backward()
                    self.optimizer_g.step()
                else:
                    test_loss = (self.output - self.gt).abs().mean()
                    self.optimizer_g.zero_grad()
                    test_loss.backward()
                    self.optimizer_g.step()
                    loss_dict['test_loss'] = test_loss
            else:
                raise RuntimeError('l_g_total is not a Tensor')

            try:
                if getattr(self, 'ema_decay', 0) > 0:
                    self.model_ema(decay=self.ema_decay)
            except Exception:
                pass

            try:
                self.log_dict = self.reduce_loss_dict(loss_dict)
            except Exception:
                self.log_dict = loss_dict

        except Exception:
            raise
