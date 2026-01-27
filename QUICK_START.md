# Pi3 Hospital Dataset Training - Quick Start Guide

## 创建的文件清单

### 1. 数据集加载器
- **`datasets/tartanair_hospital_dataset.py`** - 适配hospital数据集目录结构的数据加载器

### 2. 配置文件
- **`configs/data/tartanair_hospital.yaml`** - Hospital数据集的训练配置

### 3. 训练脚本
- **`train_hospital_from_scratch.sh`** - 完整三阶段训练脚本
- **`train_hospital_simple.sh`** - 简化单阶段训练脚本(推荐用于测试)

### 4. 测试和文档
- **`test_hospital_dataset.py`** - 数据集加载测试脚本
- **`TRAIN_HOSPITAL_README.md`** - 详细的训练说明文档
- **`QUICK_START.md`** - 本文件

## 快速开始步骤

### 步骤 1: 验证数据集

首先验证数据集路径和加载是否正常:

```bash
python test_hospital_dataset.py
```

如果看到 "Dataset Test Completed Successfully!" 就表示一切正常。

### 步骤 2: 选择训练方式

#### 方式 A: 简化训练(推荐初次测试)

只运行低分辨率训练阶段,用于快速验证:

```bash
# 单GPU
./train_hospital_simple.sh 1

# 4个GPU
./train_hospital_simple.sh 4

# 8个GPU
./train_hospital_simple.sh 8
```

#### 方式 B: 完整三阶段训练

运行完整的训练流程以获得最佳效果:

```bash
# 编辑脚本修改GPU数量
nano train_hospital_from_scratch.sh
# 修改: NUM_GPUS=8 (改为你的GPU数量)

# 运行训练
./train_hospital_from_scratch.sh
```

### 步骤 3: 监控训练

在另一个终端查看训练进度:

```bash
# 查看TensorBoard
tensorboard --logdir outputs/pi3_hospital_lowres

# 或者查看日志文件
tail -f outputs/pi3_hospital_lowres/*.log
```

### 步骤 4: 训练完成后

Checkpoints将保存在:
- 简化训练: `outputs/pi3_hospital_lowres/ckpts/`
- 完整训练:
  - Stage 1: `outputs/pi3_hospital_lowres/ckpts/`
  - Stage 2: `outputs/pi3_hospital_highres/ckpts/`
  - Stage 3: `outputs/pi3_hospital_conf/ckpts/`

## 常见问题速查

### Q1: 如何调整GPU数量?

**简化训练:**
```bash
./train_hospital_simple.sh <GPU数量>
```

**完整训练:**
编辑 `train_hospital_from_scratch.sh` 中的 `NUM_GPUS` 变量

### Q2: GPU内存不足怎么办?

编辑配置文件减少内存使用:
```bash
nano configs/train/train_pi3_lowres.yaml
```

修改:
```yaml
train:
  max_img_per_gpu: 32  # 默认64,可以减少到32或16
  resolution:
    - [196, 196]  # 可以降低到196或更小
```

### Q3: 数据集路径在哪里配置?

在 `configs/data/tartanair_hospital.yaml` 中:
```yaml
TarTanAir:
    data_root: /mnt/localssd/tartanair_tools/tartanair_data/hospital
```

### Q4: 如何从断点恢复训练?

配置中已启用自动恢复(`train.auto_resume=true`)。

使用相同的 `name` 参数重新运行命令即可:
```bash
./train_hospital_simple.sh 8
# 会自动从最新checkpoint恢复
```

### Q5: 训练需要多久?

取决于:
- GPU数量和型号
- 数据集大小 (hospital/Easy 约39个序列)
- 训练配置 (默认80 epochs)

参考时间 (8x A100 GPU):
- Stage 1 (低分辨率): 约2-4小时
- Stage 2 (高分辨率): 约1-2小时
- Stage 3 (置信度分支): 约1-2小时

### Q6: 可以只训练部分数据吗?

编辑数据集加载器或在配置中添加:
```python
# 在 configs/data/tartanair_hospital.yaml 中
TarTanAir:
    seq_num: 10  # 只使用前10个序列
```

## 目录结构

训练后的目录结构:
```
Pi3_training/
├── datasets/
│   └── tartanair_hospital_dataset.py    # 新增
├── configs/
│   └── data/
│       └── tartanair_hospital.yaml      # 新增
├── outputs/                              # 新增(训练时创建)
│   ├── pi3_hospital_lowres/
│   │   ├── ckpts/
│   │   └── tensorboard/
│   └── ...
├── train_hospital_simple.sh             # 新增
├── train_hospital_from_scratch.sh       # 新增
├── test_hospital_dataset.py             # 新增
├── TRAIN_HOSPITAL_README.md             # 新增
└── QUICK_START.md                       # 本文件
```

## 下一步

1. **验证数据集**: `python test_hospital_dataset.py`
2. **开始训练**: `./train_hospital_simple.sh 1`
3. **查看详细文档**: `cat TRAIN_HOSPITAL_README.md`

## 技术支持

- Pi3主页: https://github.com/yyfz/Pi3
- Training分支: https://github.com/yyfz/Pi3/tree/training
- 论文: https://arxiv.org/abs/2507.13347

---

**提示**: 首次训练建议先运行 `train_hospital_simple.sh 1` 用单GPU测试几个epoch,确认一切正常后再运行完整训练。
