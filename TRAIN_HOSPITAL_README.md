# Training Pi3 on TartanAir Hospital Dataset

本指南介绍如何在TartanAir Hospital数据集上从头训练Pi3模型。

## 数据集位置

数据集路径: `/mnt/localssd/tartanair_tools/tartanair_data/hospital`

数据集结构:
```
hospital/
└── Easy/
    ├── P000/
    │   ├── image_left/
    │   ├── depth_left/
    │   └── pose_left.txt
    ├── P001/
    ├── P002/
    └── ...
```

## 训练脚本

我们提供了两个训练脚本:

### 1. 完整三阶段训练 (推荐用于最佳效果)

`train_hospital_from_scratch.sh` - 完整的三阶段训练流程

**训练阶段:**
- **Stage 1**: 低分辨率训练 (224x224, 80 epochs)
- **Stage 2**: 高分辨率训练 (加载Stage 1权重)
- **Stage 3**: 置信度分支训练 (可选, 需要Segformer预训练权重)

**使用方法:**
```bash
# 编辑脚本中的GPU数量 (默认8个GPU)
# 然后运行:
./train_hospital_from_scratch.sh
```

**修改GPU数量:**
在脚本中修改 `NUM_GPUS` 变量:
```bash
NUM_GPUS=4  # 例如改为4个GPU
```

### 2. 简化单阶段训练 (快速测试)

`train_hospital_simple.sh` - 只运行低分辨率训练阶段

**使用方法:**
```bash
# 单GPU训练
./train_hospital_simple.sh

# 多GPU训练 (例如4个GPU)
./train_hospital_simple.sh 4
```

## 训练配置文件

数据配置文件: `configs/data/tartanair_hospital.yaml`

该文件已配置为:
- 仅使用TartanAir Hospital数据集
- 数据路径指向 `/mnt/localssd/tartanair_tools/tartanair_data/hospital`
- 训练时每次采样8帧图像
- z_far设置为80米

如需修改训练参数,可编辑以下配置文件:
- `configs/train/train_pi3_lowres.yaml` - 低分辨率训练参数
- `configs/train/train_pi3_highres.yaml` - 高分辨率训练参数
- `configs/train/train_pi3_conf.yaml` - 置信度分支训练参数
- `configs/model/pi3.yaml` - 模型架构配置

## 输出目录

训练完成后,checkpoints将保存在:

**完整训练:**
- Stage 1: `outputs/pi3_hospital_lowres/ckpts/`
- Stage 2: `outputs/pi3_hospital_highres/ckpts/`
- Stage 3: `outputs/pi3_hospital_conf/ckpts/`

**简化训练:**
- `outputs/pi3_hospital_lowres/ckpts/`

TensorBoard日志也会保存在对应的outputs目录中。

## 内存优化

如果遇到GPU内存不足,可以在配置文件中调整:

1. **减少每GPU图像数量** (优先级最高):
   ```yaml
   train:
     max_img_per_gpu: 32  # 默认64,可减少到32或更小
   ```

2. **减少图像分辨率**:
   ```yaml
   train:
     resolution:
       - [196, 196]  # 默认[224, 224]
   ```

3. **增加activation checkpointing**:
   ```yaml
   model:
     num_dec_blk_not_to_checkpoint: 2  # 默认4,减少此值
   ```

## 监控训练

使用TensorBoard查看训练进度:
```bash
tensorboard --logdir outputs/pi3_hospital_lowres
```

## 从checkpoint恢复训练

如果训练中断,可以从checkpoint恢复:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes 8 \
    --num_machines 1 \
    scripts/train_pi3.py \
    train=train_pi3_lowres \
    data=tartanair_hospital \
    name=pi3_hospital_lowres \
    train.resume=outputs/pi3_hospital_lowres/ckpts/checkpoint_latest.pth
```

或者设置自动恢复:
```bash
# 在配置中已经设置了 train.auto_resume=true
# 所以使用相同的name会自动从最新checkpoint恢复
```

## Stage 3 注意事项

如果要运行Stage 3 (置信度分支训练),需要:

1. 下载Segformer预训练权重:
   - 链接: https://github.com/NVlabs/SegFormer (OneDrive链接)
   - 文件名: `segformer.b0.512x512.ade.160k.pth`

2. 放置在正确位置:
   ```bash
   mkdir -p ckpts
   # 将下载的文件放到 ckpts/segformer.b0.512x512.ade.160k.pth
   ```

## 使用训练好的模型进行推理

训练完成后,可以使用checkpoint进行推理:

```python
import torch
from pi3.models.pi3 import Pi3

# 加载训练好的模型
model = Pi3.from_pretrained("path/to/checkpoint.pth")
model = model.cuda().eval()

# 进行推理
# ... (参考example.py)
```

## 常见问题

**Q: 数据集路径配置在哪里?**

A: 在 `configs/data/tartanair_hospital.yaml` 中的 `data_root` 参数。

**Q: 如何修改训练的batch size?**

A: Pi3使用动态batch size,通过 `max_img_per_gpu` 控制每个GPU的最大图像数量。

**Q: 训练需要多长时间?**

A: 取决于GPU数量和配置。使用8个GPU的话,Stage 1大约需要数小时。

**Q: 可以只在hospital数据集上训练吗?**

A: 可以,本配置文件就是专门为hospital数据集设计的。但如果数据量较小,建议考虑使用预训练权重进行fine-tune而不是从头训练。

## 技术支持

如有问题,请参考:
- Pi3主仓库: https://github.com/yyfz/Pi3
- Training分支README: https://github.com/yyfz/Pi3/tree/training
