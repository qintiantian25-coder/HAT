# 红外图像盲元修复任务使用说明

本文档说明如何使用当前项目完成红外图像盲元修复任务，包括训练、验证、测试、盲元指标评估，以及整个流程所依赖的文件与模型保存位置。

## 1. 任务目标

本项目已适配为红外图像盲元修复任务，输入为带盲元噪声的灰度图像，输出为修复后的图像。当前流程的设计目标如下：

- 训练时只保留一个最佳模型，不保存完整历史 checkpoint。
- 以验证集 PSNR 作为唯一的最佳模型判断标准。
- 当验证 PSNR 更高时，自动覆盖更新最佳模型文件。
- 验证时打印 PSNR 和 SSIM。
- 测试时输出 PSNR、SSIM、blind_mae、blind_rmse、blind_psnr 等指标。
- 所有模型、日志、验证输出和测试结果都保存在当前项目目录下的 experiments 目录中，只有数据集路径使用绝对地址。

### 1.1 环境兼容说明

- 当前项目通过 [sitecustomize.py](sitecustomize.py) 自动兼容较新的 `torchvision`，使旧版 `basicsr` 里的 `functional_tensor` 导入仍然可用。
- 因此在这个仓库根目录下运行训练/测试时，通常不需要手动修改 `basicsr` 源码或降级 `torchvision`。

## 2. 主要文件说明

下面列出本任务真正会用到的核心文件。

### 2.1 统一入口

- [main.py](main.py)
  - 统一训练和测试入口。
  - 支持以下两条命令：
    - python main.py --train --config_path ./experiment.cfg
    - python main.py --test --config_path ./experiment.cfg
  - 负责读取 experiment.cfg，并生成临时训练或测试配置。
  - 负责把训练输出写入训练日志，把测试后的盲元评估结果写入测试评估文件。

### 2.2 配置文件

- [experiment.cfg](experiment.cfg)
  - 项目主配置文件。
  - 其中数据集路径使用绝对地址：/home/student_server/Qtt/NAFNet/data_new
  - 其他所有路径都采用项目内相对路径。
  - 关键配置项包括：
    - experiments_root: ./experiments
    - model_path: ./experiments/models/best_model.pt
    - val_freq_epochs: 20
    - test_mask_root: /home/student_server/Qtt/NAFNet/data_new/test_mask

### 2.3 数据集加载

- [hat/data/blind_paired_dataset.py](hat/data/blind_paired_dataset.py)
  - 递归读取灰度图像配对数据。
  - 支持如下目录结构：
    - train_blur / train_sharp / train_mask
    - val_blur / val_sharp / val_mask
    - test_blur / test_sharp / test_mask
  - 该数据集类只使用 blur 和 sharp 作为网络输入与监督目标。
  - mask 目录不直接喂给网络，主要用于盲元专项评估。

### 2.4 模型与保存逻辑

- [hat/models/hat_model.py](hat/models/hat_model.py)
  - 模型训练、验证与最佳模型保存逻辑都在这里。
  - 验证时会根据 PSNR 自动更新最佳模型。
  - 最佳模型固定保存为 ./experiments/models/best_model.pt
  - 当 PSNR 刷新时，会打印提示信息。
  - 验证日志会写入 ./experiments/logs/validation.txt

### 2.5 测试后的盲元评估

- [tools/evaluate_blind.py](tools/evaluate_blind.py)
  - 对测试输出进行定量评估。
  - 计算全图指标：PSNR、SSIM
  - 计算盲元专项指标：blind_mae、blind_rmse、blind_psnr
  - 自动按组读取 test_mask 中的 blind_coords.csv 或 blind_pixel_coords.csv
  - 保存每张图的评估结果到 ./experiments/blind_eval/test_blind_metrics.csv

### 2.6 原项目入口

- [hat/train.py](hat/train.py)
- [hat/test.py](hat/test.py)

这两个文件仍然保留，但当前推荐使用 main.py 作为统一入口。

## 3. 数据集目录说明

你的数据集根目录为：

```bash
/home/student_server/Qtt/NAFNet/data_new
```

目录结构如下：

```text
/home/student_server/Qtt/NAFNet/data_new
├── train_blur
├── train_sharp
├── train_mask
├── val_blur
├── val_sharp
├── val_mask
├── test_blur
├── test_sharp
└── test_mask
```

### 3.1 训练集

训练时使用：

- train_blur：退化输入图像
- train_sharp：清晰真值图像

train_mask 不参与网络训练输入。

### 3.2 验证集

验证时使用：

- val_blur：验证输入图像
- val_sharp：验证真值图像

val_mask 不进入网络训练，只在需要做盲元专项分析时使用。

### 3.3 测试集

测试时使用：

- test_blur：测试输入图像
- test_sharp：测试真值图像
- test_mask：盲元坐标与掩码信息

其中 test_mask 主要用于 blind_mae、blind_rmse、blind_psnr 的计算，不作为网络输入。

## 4. 模型文件说明

本任务中最重要的模型文件是：

```text
./experiments/models/best_model.pt
```

说明如下：

- 这个文件由训练阶段自动生成。
- 它只保留验证 PSNR 最好的模型权重。
- 训练过程中如果新的验证 PSNR 更高，就会自动覆盖更新这个文件。
- 测试时会自动读取这个文件作为推理模型。

如果你重新训练，最后仍然只会保留这一份最优模型文件，不会保留大量历史 checkpoint。

## 5. 日志文件说明

所有日志都写入项目内的 experiments 目录：

```text
./experiments/logs/training.txt
./experiments/logs/validation.txt
```

### 5.1 training.txt

该文件记录完整训练过程，包括：

- 训练迭代信息
- 训练损失
- 学习率变化
- 验证触发信息
- 其他训练过程输出

### 5.2 validation.txt

该文件记录验证过程的指标，包括：

- PSNR
- SSIM
- 当前验证轮次或迭代信息
- best_model.pt 是否更新
- best_model 更新时对应的 PSNR 数值

## 6. 训练流程

### 6.1 训练指令

直接运行：

```bash
python main.py --train --config_path ./experiment.cfg
```

### 6.2 训练时流程说明

训练流程为：

1. main.py 读取 experiment.cfg。
2. main.py 根据配置生成临时训练 YAML。
3. 训练程序读取：
   - train_blur
   - train_sharp
4. 模型在训练过程中按配置的验证频率进行验证。
5. 每次验证都会计算 PSNR 和 SSIM。
6. 如果当前 PSNR 高于历史最佳值，则覆盖保存：
   - ./experiments/models/best_model.pt
7. 训练完整输出会写入：
   - ./experiments/logs/training.txt
8. 验证日志会写入：
   - ./experiments/logs/validation.txt

### 6.3 验证频率

验证频率由 experiment.cfg 中的 val_freq_epochs 控制：

```ini
val_freq_epochs = 20
```

含义是：每 20 轮验证一次。

如果你想改成每 10 轮验证一次，只需要把该值改成 10。

注意：内部实际执行时会根据训练集规模和 batch size 换算成对应的验证迭代间隔。

## 7. 测试流程

### 7.1 测试指令

直接运行：

```bash
python main.py --test --config_path ./experiment.cfg
```

### 7.2 测试时流程说明

测试流程为：

1. main.py 读取 experiment.cfg。
2. main.py 自动定位最佳模型：
   - ./experiments/models/best_model.pt
3. 测试程序读取 test_blur 和 test_sharp。
4. 模型对测试集逐张推理，结果会保存到 experiments 目录下的可视化输出目录。
5. 测试结束后自动调用 tools/evaluate_blind.py。
6. 评估脚本会计算：
   - PSNR
   - SSIM
   - blind_mae
   - blind_rmse
   - blind_psnr
7. 每张图的评估结果保存到：
   - ./experiments/blind_eval/test_blind_metrics.csv

### 7.3 测试中的盲元评估逻辑

盲元指标的计算方式与你给出的思路一致：

- 从 test_mask 中读取 blind_coords.csv 或 blind_pixel_coords.csv
- 按坐标在输出图和真值图上取像素
- 计算误差
- 汇总得到 blind_mae、blind_rmse、blind_psnr

如果同时能读到输入盲图，还会额外计算输入图对应的盲元误差，方便观察修复前后提升。

## 8. 验证流程

验证发生在训练过程中，不需要单独手动执行。

### 8.1 验证使用的数据

- val_blur
- val_sharp

### 8.2 验证输出

验证时会打印：

- PSNR
- SSIM

并把结果写入：

```text
./experiments/logs/validation.txt
```

### 8.3 最佳模型更新规则

当且仅当当前验证 PSNR 高于历史最好值时：

- 覆盖更新 ./experiments/models/best_model.pt
- 打印 best model updated 提示
- 在 validation.txt 中记录这次刷新

## 9. 推荐使用顺序

建议按下面顺序执行：

### 9.1 第一步：开始训练

```bash
python main.py --train --config_path ./experiment.cfg
```

### 9.2 第二步：查看训练日志

```text
./experiments/logs/training.txt
```

### 9.3 第三步：确认验证日志

```text
./experiments/logs/validation.txt
```

### 9.4 第四步：测试最佳模型

```bash
python main.py --test --config_path ./experiment.cfg
```

### 9.5 第五步：查看测试评估结果

```text
./experiments/blind_eval/test_blind_metrics.csv
```

## 10. 当前项目的关键约束

### 10.1 只保留一个最佳模型

项目不会保留每个 epoch 的完整历史模型，只保留：

```text
./experiments/models/best_model.pt
```

### 10.2 模型更新条件

只看验证 PSNR：

- 更高则覆盖更新
- 不更高则保持原模型不变

### 10.3 数据集路径使用绝对地址

数据集仍然使用你的绝对路径：

```text
/home/student_server/Qtt/NAFNet/data_new
```

### 10.4 其他路径全部在项目内

包括：

- 模型保存
- 日志保存
- 验证输出
- 测试评估

都在项目目录下的 experiments 里完成。

## 11. 常见修改项

如果你后续想改配置，主要改 experiment.cfg 里的这些字段：

- dataset_root：数据集根目录
- experiments_root：实验输出根目录
- model_path：测试时读取的模型文件
- val_freq_epochs：每多少轮验证一次
- test_mask_root：测试盲元坐标根目录

## 12. 一个最简执行示例

```bash
python main.py --train --config_path ./experiment.cfg
python main.py --test --config_path ./experiment.cfg
```

训练完成后，测试会自动使用：

```text
./experiments/models/best_model.pt
```

作为推理模型。
