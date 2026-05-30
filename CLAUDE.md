# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

HAT (Hybrid Attention Transformer) is a deep learning model for image super-resolution and restoration, published at CVPR 2023. It's built on [BasicSR](https://github.com/XPixelGroup/BasicSR) v1.3.4.9 and extends it with custom architectures, models, and datasets.

## Commands

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```

### Standard super-resolution testing
```bash
python hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```

### Standard super-resolution training (distributed)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 --master_port=4321 hat/train.py \
  -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
```

### Blind restoration (unified entrypoint)
```bash
python main.py --train --config_path ./experiment.cfg
python main.py --test --config_path ./experiment.cfg
```

`main.py` reads the INI-style `experiment.cfg`, programmatically builds a YAML config, writes it to a temp file, and delegates to `hat/train.py` or `hat/test.py`. It also runs `tools/evaluate_blind.py` automatically after testing to compute blind-pixel metrics.

### Blind evaluation only
```bash
python tools/evaluate_blind.py --out_dir /path/to/outputs --gt_dir /path/to/gts \
  --input_dir /path/to/test_blur --mask_root /path/to/test_mask --save_dir /path/to/save
```

## Architecture

### Entry points

- **`hat/train.py`** / **`hat/test.py`**: Thin wrappers that import `hat.*` subpackages (triggering registry auto-discovery) then call `basicsr.train.train_pipeline` / `basicsr.test.test_pipeline`.
- **`main.py`**: Higher-level entrypoint for blind restoration. Parses `experiment.cfg`, generates temporary YAML, launches training/testing as a subprocess, and runs blind-pixel evaluation afterward.

### Package structure (`hat/`)

All subpackages use BasicSR's registry pattern: `__init__.py` auto-scans modules matching `*_arch.py`, `*_model.py`, `*_dataset.py` and imports them — this causes `@ARCH_REGISTRY.register()`, `@MODEL_REGISTRY.register()`, `@DATASET_REGISTRY.register()` decorators to fire.

- **`hat/archs/`**: Network architectures registered in BasicSR's `ARCH_REGISTRY`.
  - `hat_arch.py` — HAT (main model): hybrid attention with channel attention blocks (CAB), window-based self-attention (HAB), and overlapping cross-attention (OCAB). Supports gradient checkpointing and optional PixelShuffle upsampling.
  - `discriminator_arch.py` — `UNetDiscriminatorSN` for GAN training.
  - `srvgg_arch.py` — `SRVGGNetCompact`, a lightweight VGG-style SR network.

- **`hat/models/`**: Training/inference logic registered in `MODEL_REGISTRY`.
  - `hat_model.py` — `HATModel` (extends `SRModel`). Key overrides:
    - `optimize_parameters()` is fully self-contained — computes forward pass + L1 loss + backward, does NOT call `super()`.
    - `save()` / `save_network()` / `save_training_state()` are no-ops — only best-checkpoint saving via `_update_best_metric_result()`.
    - `_update_best_metric_result()` saves `best_model.pt` under `experiments/models/` when PSNR improves (overwrites on improvement), with metadata to `best_model_meta.json`. Prints a log message on update.
    - `nondist_validation()` handles window-aligned padding, optional tile processing, computes full-image PSNR/SSIM via `hat.utils.metric_utils`, AND computes blind-pixel metrics (MAE/RMSE/PSNR) at coordinates from `blind_coords.csv` in the mask directory. All metric values are printed and written to `experiments/logs/validation.txt`.
  - `realhatgan_model.py` — `RealHATGANModel` (GAN-based, extends `SRGANModel`).
  - `realhatmse_model.py` — `RealHATMSEModel` (MSE-based, extends `SRModel`).

- **`hat/data/`**: Datasets registered in `DATASET_REGISTRY`.
  - `blind_paired_dataset.py` — `BlindPairedImageDataset`: recursive paired PNGs in grouped directory layout (`lq/group/img.png`, `gt/group/img.png`).
  - `imagenet_paired_dataset.py` — `ImageNetPairedDataset`: standard SR paired data with bicubic downsampling.
  - `realesrgan_dataset.py` — `RealESRGANDataset`: degradation pipeline for real-world SR training.

- **`hat/utils/metric_utils.py`**: Standalone PSNR/SSIM functions operating on numpy uint8 arrays.

### Configuration system

There are two config formats:
1. **YAML** (BasicSR-native): Used by `hat/train.py` and `hat/test.py`. Configs in `options/train/` and `options/test/`.
2. **INI** (`experiment.cfg`): Used by `main.py`. Simpler flat format with `[common]`, `[paths]`, `[train]`, `[test]`, `[logger]` sections. `main.py` converts these into the full YAML structure at runtime.

### Compatibility shim

`sitecustomize.py` creates a shim module `torchvision.transforms.functional_tensor` that redirects to the modern `torchvision.transforms.functional`. This is loaded automatically by Python (it's in the repo root) and keeps older BasicSR imports working on newer torchvision.

## Blind pixel dataset structure

The blind pixel dataset is stored under `data/` (root directory). All images are grayscale, 640×512 pixels, lossless PNG format, and sequentially numbered (1.png, 2.png, ...) in natural temporal order.

### Split organization

| Split | Groups | Purpose |
|-------|--------|---------|
| train | 001–007 | `train_blur/`, `train_sharp/`, `train_mask/` |
| val | 001–002 | `val_blur/`, `val_sharp/`, `val_mask/` |
| test | 001–006 | `test_blur/`, `test_sharp/`, `test_mask/` |

### Per-group structure

Each group directory (e.g., `train_blur/001/`) contains a sequence of identically named PNGs. The three modalities are aligned across corresponding group directories:

- **`blur/`** — Degraded input images with simulated blind-pixel noise (sensor simulation)
- **`sharp/`** — Clean ground truth, temporally and spatially aligned to blur
- **`mask/`** — Blind-pixel prior annotations:
  - `blind_coords.csv` — Static blind pixel coordinates (columns: x, y). Records inherent sensor defects.
  - `flash_pixel_coords.csv` — Frame-level flash pixel records capturing dynamic temporal response anomalies.
  - `blind_pixel_mask.png` — Binary mask image at input resolution (640×512), used as an additional channel for guiding network spatial feature concatenation.

The temporal sequence, spatial resolution, and mask indexing are precisely aligned across all three modalities within each group, enabling multi-modal sequence loading via DataLoader.

### Configuration paths

`experiment.cfg` `[paths]` section maps to these data directories. The `[common]` section's `dataset_root` defaults to `/home/student_server/Qtt/NAFNet/data_new` but should be pointed to the local `data/` directory for actual use.

## Key constraints

- Avoid PyTorch 1.8 (causes abnormal performance per README).
- The HAT model requires input dimensions divisible by `window_size` (default 16); validation code auto-pads.
- Training is GPU-memory intensive (~20GB/GPU with batch_size=4). Default settings for blind restoration use `batch_size_per_gpu=1` with `gradient_accumulation_steps=4` and `use_checkpoint=true`.
- `HATModel.optimize_parameters()` does not use the standard BasicSR loss pipeline — it has its own self-contained forward+backward logic.
- **Blind pixel restoration specific**: Input is 2-channel (blur + mask concatenated), output is 1-channel (restored grayscale). HAT supports this via `in_chans=2, out_chans=1`. The mask PNG is a per-group static sensor defect pattern that tells the network where blind pixels are.
- Validation uses PSNR for best model selection; validation frequency is controlled by `val_freq_epochs` in `experiment.cfg` (default 20 epochs).
- Only one best model is kept at `experiments/models/best_model.pt` — no per-epoch checkpoints.
