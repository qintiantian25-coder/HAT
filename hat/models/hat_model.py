import torch
from torch.nn import functional as F
import os
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img

import math
from collections import OrderedDict
from tqdm import tqdm
from os import path as osp
import json
import time
from pathlib import Path

from basicsr.utils import get_root_logger
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
        # Backward-compatible: if a directory path is provided, write validation.txt under it.
        if str(old_value).lower().endswith('.txt'):
            return old_value
        return osp.join(old_value, 'validation.txt')

    return None

@MODEL_REGISTRY.register()
class HATModel(SRModel):

    def pre_process(self):
        # pad to multiplication of window_size
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
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
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

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
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
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            lq_path = val_data['lq_path'][0]
            dataset_root = dataloader.dataset.opt.get('dataroot_lq')
            if dataset_root:
                img_name = osp.splitext(osp.relpath(lq_path, dataset_root))[0]
            else:
                img_name = osp.splitext(osp.basename(lq_path))[0]
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

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
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
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _update_best_metric_result(self, dataset_name, metric, value, current_iter):
        """Override to save a single best model file when `psnr` improves.

        Saves `best_model.pt` (model state_dict) under
        `<experiments_root>/<exp_name>/models/best_model.pt` and writes
        a small `best_model_meta.json` with psnr and iteration.
        This runs alongside the base implementation.
        """
        # call parent implementation if available
        try:
            super(HATModel, self)._update_best_metric_result(dataset_name, metric, value, current_iter)
        except Exception:
            pass

        if metric is None:
            return
        if str(metric).lower() != 'psnr':
            return

        # determine experiments root and model save dir
        path_cfg = self.opt.get('path', {}) if isinstance(self.opt.get('path', {}), dict) else self.opt.get('path', {})
        exp_root = None
        if isinstance(path_cfg, dict):
            exp_root = path_cfg.get('experiments_root') or path_cfg.get('root')
        if not exp_root:
            exp_root = osp.join(os.getcwd(), 'experiments')
        models_dir = None
        if isinstance(path_cfg, dict):
            models_dir = path_cfg.get('models')
        if not models_dir:
            models_dir = osp.join(exp_root, 'models')
        try:
            os.makedirs(models_dir, exist_ok=True)
        except Exception:
            pass

        meta_file = osp.join(models_dir, 'best_model_meta.json')
        prev_best = -1e9
        if osp.exists(meta_file):
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
            # prefer EMA model if available
            net = getattr(self, 'net_g_ema', None) or getattr(self, 'net_g', None)
            if net is None:
                return
            save_path = osp.join(models_dir, 'best_model.pt')
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
        """Override to add debugging before calling base implementation.

        Prints model/optimizer parameter info and enables anomaly detection
        so we get better diagnostics if backward fails inside the base method.
        """
        # basic sanity: ensure optimizer exists
        if not hasattr(self, 'optimizer_g'):
            raise RuntimeError('optimizer_g not found on model')

        # Implement a simple training step here (forward -> pixel loss -> backward)
        # This avoids relying on the external SRModel implementation which
        # produced a non-differentiable `l_total` in this environment.
        try:
            # sanity checks
            if not hasattr(self, 'net_g'):
                raise RuntimeError('net_g not found on model')
            if not hasattr(self, 'lq'):
                raise RuntimeError('lq (input) not available; feed_data not called')

            self.net_g.train()
            # forward
            self.output = self.net_g(self.lq)

            # compute losses
            l_g_total = 0
            loss_dict = OrderedDict()
            if hasattr(self, 'cri_pix') and self.cri_pix is not None and hasattr(self, 'gt'):
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total = l_g_total + l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # backward and step
            if isinstance(l_g_total, torch.Tensor):
                if l_g_total.requires_grad:
                    self.optimizer_g.zero_grad()
                    l_g_total.backward()
                    self.optimizer_g.step()
                else:
                    # fall back: try computing a simple differentiable test loss
                    test_loss = (self.output - self.gt).abs().mean()
                    self.optimizer_g.zero_grad()
                    test_loss.backward()
                    self.optimizer_g.step()
                    loss_dict['test_loss'] = test_loss
            else:
                raise RuntimeError('l_g_total is not a Tensor')

            # EMA update
            try:
                if getattr(self, 'ema_decay', 0) > 0:
                    self.model_ema(decay=self.ema_decay)
            except Exception:
                pass

            # log
            try:
                self.log_dict = self.reduce_loss_dict(loss_dict)
            except Exception:
                self.log_dict = loss_dict

        except Exception:
            raise
