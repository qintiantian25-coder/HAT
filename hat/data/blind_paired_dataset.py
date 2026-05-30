import os
import os.path as osp
import re

import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def collect_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith('.png'):
                paths.append(osp.join(root, file_name))
    return sorted(paths, key=natural_sort_key)


@DATASET_REGISTRY.register()
class BlindPairedImageDataset(data.Dataset):
    """Recursive paired dataset for blind restoration.

    Expected layout:
      dataroot_lq/group_id/image.png
      dataroot_gt/group_id/image.png
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['dataroot_lq']
        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        lq_paths = collect_images(self.lq_folder)
        gt_paths = collect_images(self.gt_folder)

        lq_map = {osp.relpath(p, self.lq_folder): p for p in lq_paths}
        gt_map = {osp.relpath(p, self.gt_folder): p for p in gt_paths}
        common_keys = sorted(set(lq_map.keys()) & set(gt_map.keys()), key=natural_sort_key)
        self.paths = [(lq_map[k], gt_map[k]) for k in common_keys]

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f'No paired PNG files found under {self.lq_folder} and {self.gt_folder}. '
                'Check the recursive group layout.'
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path, gt_path = self.paths[index]
        lq_bytes = self.file_client.get(lq_path, 'lq')
        gt_bytes = self.file_client.get(gt_path, 'gt')

        img_lq = imfrombytes(lq_bytes, flag='grayscale', float32=True)
        img_gt = imfrombytes(gt_bytes, flag='grayscale', float32=True)

        if img_lq.ndim == 2:
            img_lq = img_lq[:, :, None]
        if img_gt.ndim == 2:
            img_gt = img_gt[:, :, None]

        scale = self.opt['scale']

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        else:
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
