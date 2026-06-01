# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

HAT (Hybrid Attention Transformer) is a deep learning model for image super-resolution and restoration, published at CVPR 2023. It's built on [BasicSR](https://github.com/XPixelGroup/BasicSR) v1.3.4.9 and extends it with custom architectures, models, and datasets.

There are **no unit tests** in this repository. Testing means running inference on benchmark datasets.

## Commands

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```

### Linting
```bash
flake8                              # max-line-length=120, ignores W503/W504
yapf -r -d .                       # pep8 style, column_limit=120
isort --check-only .                # line_length=120
```
Config is in `setup.cfg`.

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

`main.py` reads the INI-style `experiment.cfg`, programmatically builds a YAML config, writes it to a temp file, and delegates to `hat/train.py` or `hat/test.py` as a subprocess. After testing, it automatically runs `tools/evaluate_blind.py` to compute blind-pixel metrics. During `--test`, if `best_model.pt` is not found, it auto-falls-back to `net_g_latest.pth` if available.

### Blind evaluation only
```bash
python tools/evaluate_blind.py --out_dir /path/to/outputs --gt_dir /path/to/gts \
  --input_dir /path/to/test_blur --mask_root /path/to/test_mask --save_dir /path/to/save
```

## Architecture

### Entry points

- **`hat/train.py`** / **`hat/test.py`**: Thin wrappers that import `hat.*` subpackages (triggering registry auto-discovery) then call `basicsr.train.train_pipeline` / `basicsr.test.test_pipeline`.
- **`main.py`**: Higher-level entrypoint for blind restoration. Parses `experiment.cfg`, generates temporary YAML, launches training/testing as a subprocess, and runs blind-pixel evaluation afterward.
- **`predict.py`**: Cog/Replicate predictor for deployment. Copies an input image, runs `hat/test.py` with `HAT_SRx4_ImageNet-LR.yml`, returns the output PNG.

### Package structure (`hat/`)

All subpackages use BasicSR's registry pattern: `__init__.py` auto-scans modules matching `*_arch.py`, `*_model.py`, `*_dataset.py` and imports them — this causes `@ARCH_REGISTRY.register()`, `@MODEL_REGISTRY.register()`, `@DATASET_REGISTRY.register()` decorators to fire.

- **`hat/archs/`**: Network architectures.
  - `hat_arch.py` — HAT: hybrid attention with channel attention blocks (CAB), window-based self-attention (HAB), and overlapping cross-attention (OCAB). Supports gradient checkpointing and optional PixelShuffle upsampling.
  - `discriminator_arch.py` — `UNetDiscriminatorSN` for GAN training.
  - `srvgg_arch.py` — `SRVGGNetCompact`, a lightweight VGG-style SR network.

- **`hat/models/`**: Training/inference logic.
  - `hat_model.py` — `HATModel` (extends `SRModel`). Key behaviors:
    - `optimize_parameters()` is fully self-contained — forward pass + L1 loss + backward, does NOT call `super()`. If `l_g_total.requires_grad` is False (no L1 loss configured), it falls back to a `test_loss` based on `(output - gt).abs().mean()` so training still works.
    - `save()` / `save_network()` / `save_training_state()` are **no-ops** — all standard BasicSR checkpointing is suppressed. Training **cannot be resumed** (no optimizer/scheduler/iteration state saved).
    - `_update_best_metric_result()` saves `net_g_ema.state_dict()` (or `net_g.state_dict()`) to `experiments/models/best_model.pt` when PSNR improves, overwriting the previous best. Metadata written to `best_model_meta.json`. The save path is hardcoded to `experiments/models/` — it does NOT use the config's `path.models` setting.
    - `nondist_validation()` handles window-aligned padding, optional tile processing, full-image PSNR/SSIM via `hat.utils.metric_utils`, AND blind-pixel metrics (MAE/RMSE/PSNR) at coordinates from `blind_coords.csv`. All values printed and appended to `experiments/logs/validation.txt`.
  - `realhatgan_model.py` — `RealHATGANModel` (extends `SRGANModel`). Used for GAN-based real-world SR with perceptual + adversarial losses.
  - `realhatmse_model.py` — `RealHATMSEModel` (extends `SRModel`). MSE-pretraining stage before GAN finetuning.

- **`hat/data/`**: Datasets.
  - `blind_paired_dataset.py` — `BlindPairedImageDataset`: recursive paired PNGs in grouped directory layout.
  - `imagenet_paired_dataset.py` — `ImageNetPairedDataset`: standard SR paired data with bicubic downsampling.
  - `realesrgan_dataset.py` — `RealESRGANDataset`: degradation pipeline for real-world SR training.

- **`hat/utils/metric_utils.py`**: Standalone PSNR/SSIM functions on numpy uint8 arrays.

### HAT model variants

The HAT architecture is parameterized; the three size variants differ in embedding dimension and attention compression:

| Variant | Params | `embed_dim` | `compress_ratio` | `squeeze_factor` |
|---------|--------|-------------|-------------------|-------------------|
| HAT-S   | 9.6M   | 144         | 24                | 24                |
| HAT     | 20.8M  | 180         | 3                 | 30                |
| HAT-L   | larger  | (larger)    | 3                 | 30                |

All share `depths=[6,6,6,6,6,6]`, `num_heads=[6,6,6,6,6,6]`, `window_size=16`, `img_size=64`. Configs for each are in `options/train/` and `options/test/`.

### Configuration system

Two config formats coexist:

1. **YAML** (BasicSR-native): Used directly by `hat/train.py` and `hat/test.py`. Configs in `options/train/` and `options/test/`.
2. **INI** (`experiment.cfg`): Used by `main.py` for blind restoration. Flat format with `[common]`, `[paths]`, `[train]`, `[val]`, `[test]`, `[logger]` sections. `main.py` converts INI → nested dict → temp YAML file → passed to `hat/train.py` or `hat/test.py`. The `[common].dataset_root` defaults to `/home/student_server/Qtt/NAFNet/data_new` — point it to the local `data/` directory for actual use.

### Compatibility shim

`sitecustomize.py` creates a fake module `torchvision.transforms.functional_tensor` that redirects all attributes to the modern `torchvision.transforms.functional`. Python auto-loads `sitecustomize.py` from the working directory, so no explicit import is needed. This keeps legacy BasicSR code (`from torchvision.transforms.functional_tensor import rgb_to_grayscale`) working on newer torchvision versions.

## Blind pixel dataset structure

The blind pixel dataset uses a grouped directory layout. All images are grayscale, 640×512 pixels, lossless PNG, sequentially numbered (1.png, 2.png, ...).

### Split organization

| Split | Groups | Directories |
|-------|--------|-------------|
| train | 001–007 | `train_blur/`, `train_sharp/`, `train_mask/` |
| val   | 001–002 | `val_blur/`, `val_sharp/`, `val_mask/` |
| test  | 001–006 | `test_blur/`, `test_sharp/`, `test_mask/` |

### Per-group contents

Each group directory (e.g., `train_blur/001/`) contains a sequence of identically named PNGs aligned across modalities:

- **`blur/`** — Degraded input images with simulated blind-pixel noise.
- **`sharp/`** — Clean ground truth, temporally and spatially aligned.
- **`mask/`** — Blind-pixel prior annotations:
  - `blind_coords.csv` — Static blind pixel coordinates (columns: `x, y`).
  - `flash_pixel_coords.csv` — Frame-level flash pixel records.
  - `blind_pixel_mask.png` — Binary mask at input resolution (640×512), concatenated as a second input channel.

### Blind pixel metric workflow

During validation/testing, for each image the LQ path is mirrored under `mask_root` to find the corresponding `blind_coords.csv`. Pixel values at those (x,y) coordinates are sampled from both the output and GT, then MAE/RMSE/PSNR are computed over only those blind pixels.

## Key constraints

- **PyTorch 1.8 is broken** — causes abnormal performance per README. Use ≥1.7 but not 1.8.
- HAT requires input dimensions divisible by `window_size` (default 16); validation code auto-pads with reflect mode.
- Training is GPU-memory intensive (~20GB/GPU with batch_size=4). Blind restoration defaults use `batch_size_per_gpu=1`, `gradient_accumulation_steps=4`, and `use_checkpoint=true`.
- **No training resumption**: `HATModel` disables all BasicSR checkpoint saving. Only the single best PSNR model is kept at `experiments/models/best_model.pt`. No optimizer state, scheduler state, or iteration counter is saved.
- `HATModel.optimize_parameters()` bypasses the standard BasicSR loss pipeline entirely — it only supports pixel-wise L1 loss. For perceptual/GAN losses, use `RealHATGANModel`.
- **Blind pixel restoration**: Input is 2-channel (blur + mask concatenated via `in_chans=2`), output is 1-channel (`out_chans=1`). Scale is 1 (no upsampling).
- Validation uses PSNR for best model selection; frequency controlled by `val_freq_epochs` in `experiment.cfg` (default 20). `main.py` converts this to iterations via `ceil(train_images / effective_batch) * val_freq_epochs`.
