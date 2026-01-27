# Pi3 预训练模型深度可视化指南

本文档说明如何使用 `visualize_depth.py` 可视化 Pi3 预训练模型的深度预测，并对深度进行缩放。

## 主要功能

1. **支持预训练模型**: 可以加载从网上下载的 Pi3 预训练检查点
2. **深度缩放**: 自动计算并应用最优缩放因子，将预测深度对齐到真实深度范围
3. **对比可视化**: 显示 RGB、真实深度、原始预测深度和缩放后的预测深度

## 使用方法

### 1. 使用预训练模型（从网上下载）

```bash
python visualize_depth.py \
    --pretrained_ckpt /path/to/pretrained_checkpoint.pth \
    --data_root /path/to/tartanair/hospital \
    --output_dir ./outputs/pretrained_visualizations \
    --num_samples 10 \
    --frame_num 8 \
    --scale_depth \
    --device cuda
```

### 2. 使用训练好的本地模型

```bash
python visualize_depth.py \
    --ckpt_dir /path/to/trained/model/dir \
    --data_root /path/to/tartanair/hospital \
    --output_dir ./outputs/trained_visualizations \
    --num_samples 10 \
    --frame_num 8 \
    --scale_depth \
    --device cuda
```

### 3. 使用提供的 Shell 脚本

修改 `visualize_pretrained.sh` 中的路径参数后运行：

```bash
chmod +x visualize_pretrained.sh
./visualize_pretrained.sh
```

## 命令行参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--pretrained_ckpt` | str | 预训练检查点文件路径（从网上下载） | None |
| `--ckpt_dir` | str | 训练好的模型目录路径 | 见脚本 |
| `--data_root` | str | TartanAir Hospital 数据集根目录 | 见脚本 |
| `--output_dir` | str | 可视化结果保存目录 | 见脚本 |
| `--num_samples` | int | 要可视化的样本数量 | 5 |
| `--frame_num` | int | 每个序列使用的帧数 | 8 |
| `--scale_depth` | flag | 是否对预测深度进行缩放 | False |
| `--device` | str | 推理设备 (cuda/cpu) | cuda |

## 深度缩放原理

使用最小二乘法计算最优缩放因子：

```
scale = sum(depth_gt * depth_pred) / sum(depth_pred^2)
depth_scaled = depth_pred * scale
```

这个缩放因子可以将预测深度的范围对齐到真实深度范围，使得两者具有相同的尺度。

## 输出说明

### 生成的可视化图像

每个样本会生成包含 4 列的对比图：

1. **RGB Image**: 原始 RGB 图像
2. **Ground Truth Depth**: 真实深度图
3. **Predicted Depth (unscaled)**: 未缩放的预测深度图（显示原始预测范围）
4. **Predicted Depth (scaled)**: 缩放后的预测深度图（与真实深度同一尺度）

### 缩放统计信息

如果使用 `--scale_depth` 参数，会在控制台输出统计信息：

```
Depth scaling statistics:
  Mean scale: 15.3421
  Median scale: 15.1234
  Std scale: 2.3456
  Min scale: 12.4567
  Max scale: 18.9012
```

这些统计数据可以帮助了解预训练模型的深度预测尺度特性。

## 示例

假设你从 HuggingFace 下载了 Pi3 预训练模型到 `ckpts/pi3_pretrained.pth`：

```bash
python visualize_depth.py \
    --pretrained_ckpt ckpts/pi3_pretrained.pth \
    --data_root /mnt/localssd/tartanair_tools/tartanair_data/hospital \
    --output_dir ./outputs/pi3_pretrained \
    --num_samples 20 \
    --frame_num 8 \
    --scale_depth \
    --device cuda
```

结果将保存在 `./outputs/pi3_pretrained/depth_visualizations/` 目录中。

## 注意事项

1. **内存要求**: 使用较大的 `frame_num` 和 `num_samples` 时需要更多 GPU 内存
2. **预训练模型格式**: 确保预训练检查点是 `.pth` 或 `.bin` 格式的 PyTorch 模型
3. **数据集路径**: TartanAir Hospital 数据集必须包含 RGB 图像和深度图
4. **深度缩放**: 建议始终使用 `--scale_depth` 来更好地评估模型的深度预测质量

## 相关文件

- `visualize_depth.py` - 主要可视化脚本
- `visualize_pretrained.sh` - 使用预训练模型的示例脚本
- `visualize_depth_pretrained.py` - 另一个预训练模型可视化版本
