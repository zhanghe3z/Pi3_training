# Sigma_Z2 计算和可视化修复总结

## 问题描述
训练时出现错误：`Error computing sigma_z2: a Tensor with 3 elements cannot be converted to Scalar`

## 根本原因
在 `trainers/pi3_trainer.py` 的 `compute_sigma_z2_from_view` 方法中：
- `view['camera_intrinsics']` 经过数据 collate 后形状为 `(B, 3, 3)`（包含batch维度）
- 代码直接访问 `K[0, 0].item()`，实际上获取的是整个第一行 `(3,)` 而不是标量
- `.item()` 无法将包含3个元素的tensor转换为标量，导致报错

## 修复内容

### 1. 修复 camera_intrinsics 索引问题
**文件**: `trainers/pi3_trainer.py:274-297`

```python
# 修复前
K = view['camera_intrinsics']  # (3, 3) numpy array
if isinstance(K, np.ndarray):
    K = torch.from_numpy(K).to(depth_gt.device)
fx = K[0, 0].item()  # ❌ 错误：K可能是(B, 3, 3)，K[0,0]返回(3,)

# 修复后
K = view['camera_intrinsics']  # Could be (B, 3, 3) or (3, 3)
if isinstance(K, np.ndarray):
    K = torch.from_numpy(K).to(depth_gt.device)
# If batched, select the first sample
if K.ndim == 3:
    K = K[0]  # (B, 3, 3) -> (3, 3)
fx = K[0, 0].item()  # ✅ 正确：K现在是(3, 3)，K[0,0]是标量
```

### 2. 改进 variance 伪彩色可视化
**文件**: `trainers/pi3_trainer.py:245-257`

**改进点**：
- ✅ 使用 `turbo` colormap 替代 `viridis`，提供更明显的伪彩色效果
- ✅ 添加更多统计信息：min, max, mean, std
- ✅ 使用科学计数法显示 variance 值（更易读）
- ✅ 保持 log scale 可视化以展示更大的动态范围

```python
# 改进后的可视化
im2 = axes[1].imshow(sigma_z2_vis, cmap='turbo')  # 使用 turbo 伪彩色
sigma_mean = sigma_z2[valid_gt].mean() if valid_gt.any() else 0
sigma_std = sigma_z2[valid_gt].std() if valid_gt.any() else 0
sigma_min = sigma_z2[valid_gt].min() if valid_gt.any() else 0
sigma_max = sigma_z2[valid_gt].max() if valid_gt.any() else 0
axes[1].set_title(f'log10(σ²_Z) - Depth Variance\n'
                  f'μ={sigma_mean:.2e} σ={sigma_std:.2e}\n'
                  f'min={sigma_min:.2e} max={sigma_max:.2e}')
```

## 验证测试

运行测试脚本 `test_sigma_z2_fix.py` 验证修复：
```bash
python test_sigma_z2_fix.py
```

**测试结果**：
✅ Intrinsics extraction test passed!
✅ Visualization saved to test_sigma_z2_fix_visualization.png
✅ All tests passed!

## 可视化效果

测试生成的图片展示了三个面板：
1. **Ground Truth Depth**: 原始深度图
2. **σ²_Z (raw)**: 原始 variance 值
3. **log10(σ²_Z) - Pseudo-color**: 使用 turbo 伪彩色映射的 log-scale variance

**观察结果**：
- ✅ 在深度不连续处（边缘）variance 较高（红色/黄色）
- ✅ 在平坦区域 variance 较低（蓝色）
- ✅ 伪彩色清晰显示了 variance 的空间分布

## 下一步

修复已完成，可以重新运行训练：
```bash
bash train_hospital_local_points_gt_pred.sh
```

现在 sigma_z2 计算应该能正常工作，并且 WandB 中会显示正确的 variance 伪彩色可视化。
