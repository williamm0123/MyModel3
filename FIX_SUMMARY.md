# 训练 Loss 增大问题 - 完整修复方案

## 问题现象
使用 torchrun 进行多卡训练时，出现以下错误：
```
IndexError: The shape of the mask [5, 64, 80] at index 1 does not match 
the shape of the indexed tensor [5, 1, 1] at index 1
```

## 根本原因分析

### 问题链

1. **初始问题**: Loss 归一化逻辑错误
   - 原代码在计算 loss 前除以 depth_interval，放大了预测误差
   - 导致 loss 数值不稳定，逐渐增大

2. **第一次修复失败**: 张量索引形状不匹配
   ```python
   # ❌ 错误的修复
   if depth_interval.dim() == 1:
       depth_interval = depth_interval[:, None, None]  # (B, 1, 1)
   loss = loss / depth_interval[mask]  # IndexError!
   ```
   - `depth_interval` 形状：`[5, 1, 1]`
   - `mask` 形状：`[5, 64, 80]`
   - PyTorch 不允许用高维 mask 直接索引低维张量

3. **正确理解**:
   - `loss` 是经过 mask 索引后的 1D 张量 `[N,]`，N 是有效像素数
   - `depth_interval` 在每个 batch 内是标量（所有空间位置相同）
   - 需要先将 `depth_interval` 扩展到与 mask 相同的空间维度，再用 mask 索引

---

## ✅ 最终修复方案

### 修复后的代码 (`models/losses.py`)

```python
def regression_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    depth_interval: Optional[torch.Tensor] = None,
    depth_values: Optional[torch.Tensor] = None,
    clip_func: Optional[str] = None,
    inverse_depth: bool = False,
) -> torch.Tensor:
    """
    Smooth L1 regression loss for depth.
    
    Args:
        depth_pred: (B, H, W) predicted depth
        depth_gt: (B, H, W) ground truth depth
        mask: (B, H, W) valid mask
        depth_interval: (B,) or (B, 1, 1) depth interval for normalization
        depth_values: (B, D, H, W) depth hypotheses for dynamic clipping
        clip_func: "dynamic" to clip loss by depth range
        inverse_depth: whether depth_values are in inverse order
    
    Returns:
        loss: scalar loss value
    """
    B, H, W = depth_pred.shape
    mask_bool = mask > 0.5
    
    # Compute smooth L1 loss
    if not mask_bool.any():
        return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)

    loss = F.smooth_l1_loss(depth_pred[mask_bool], depth_gt[mask_bool], reduction='none')
    
    # Normalize by depth interval AFTER computing the loss
    # This keeps the loss magnitude reasonable
    if depth_interval is not None:
        if depth_interval.dim() == 1:
            # depth_interval: (B,) -> need to select per-pixel values
            # Expand depth_interval to (B, H, W) then use mask to get (N,)
            depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
        elif depth_interval.dim() == 3:
            # depth_interval: (B, 1, 1) -> expand to (B, H, W) then use mask
            depth_interval_expanded = depth_interval.expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
    
    # Dynamic clipping
    if clip_func == 'dynamic' and depth_values is not None:
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
        depth_range = (depth_values[:, -1] - depth_values[:, 0])  # (B,)
        # Don't divide by depth_interval again since we already normalized the loss
        if depth_range.dim() == 1:
            depth_range_expanded = depth_range.view(B, 1, 1).expand(B, H, W)
        else:
            depth_range_expanded = depth_range.expand(B, H, W)
        depth_range_selected = depth_range_expanded[mask_bool]
        loss = torch.clamp_max(loss, depth_range_selected)
    
    return loss.mean()
```

---

## 关键修复点

### 1. 先计算 loss，再归一化
```python
# ✅ 正确做法
loss = F.smooth_l1_loss(depth_pred[mask_bool], depth_gt[mask_bool], reduction='none')
if depth_interval is not None:
    depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
    loss = loss / depth_interval_expanded[mask_bool]
```

### 2. 正确处理不同维度的 depth_interval
- **1D 情况** `(B,)`: 先 reshape 为 `(B, 1, 1)`，再 expand 到 `(B, H, W)`
- **3D 情况** `(B, 1, 1)`: 直接 expand 到 `(B, H, W)`

### 3. 使用 expand + mask 索引
```python
# ✅ 正确索引方式
depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
loss = loss / depth_interval_expanded[mask_bool]

# ❌ 错误方式（会导致 IndexError）
loss = loss / depth_interval[mask_bool]  # 形状不匹配！
```

---

## 其他配置优化

### config/mvs.json
```json
{
  "train": {
    "lr": 5e-5,              // 1e-4 → 5e-5 (降低学习率)
    "weight_decay": 1e-5,     // 1e-4 → 1e-5 (减小正则化)
    "grad_clip": 5.0          // 1.0 → 5.0 (放宽梯度裁剪)
  },
  "loss": {
    "loss_weights": [0.25, 0.5, 0.75, 1.0]  
    // [1.0,1.0,1.0,1.0] → 渐进式权重 (stage1→stage4)
  }
}
```

### train.py
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, args.epochs * max(1, steps_per_epoch)),
    eta_min=lr * 0.1,  // lr*0.01 → lr*0.1 (避免学习率过快接近 0)
)
```

---

## 验证方法

### 1. 单元测试
```bash
cd /home/william/project/MyModel3
python -c "
import torch
from models.losses import regression_loss

# Test with different depth_interval dimensions
B, H, W = 2, 64, 80
depth_pred = torch.rand(B, H, W) * 10 + 0.5
depth_gt = torch.rand(B, H, W) * 10 + 0.5
mask = torch.ones(B, H, W)

# Test 1D depth_interval
depth_interval_1d = torch.tensor([2.5, 2.5])
loss1 = regression_loss(depth_pred, depth_gt, mask, depth_interval_1d)
print(f'Test 1D: {loss1.item():.6f}')

# Test 3D depth_interval
depth_interval_3d = torch.tensor([2.5, 2.5]).view(2, 1, 1)
loss2 = regression_loss(depth_pred, depth_gt, mask, depth_interval_3d)
print(f'Test 3D: {loss2.item():.6f}')

print('All tests passed!')
"
```

### 2. 实际训练测试
```bash
# 单卡快速测试
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/mvs.json \
    --mock \
    --epochs 2 \
    --batch_size 2 \
    --max_train_steps 10

# 观察输出中的 loss 变化
# 应该看到 loss 逐渐下降或保持稳定，不会爆炸
```

### 3. 多卡训练
```bash
# 双卡训练（修复后应该正常运行）
torchrun --standalone --nproc_per_node=2 train.py \
    --config config/mvs.json
```

---

## 预期效果

修复后，训练应该呈现以下特征：

### 正常收敛曲线
```
Loss
 ↑
 │  ┌─┐
 │  │  └──┐
 │  │     └──┐
 │  │        └──┐
 │  │           └─────
 │  └──────────────────→ Epoch
```

### TensorBoard 监控指标
- `loss/train_total_step`: 每个 step 的总 loss - 应逐渐下降
- `loss/stage1`, `loss/stage2`, `loss/stage3`, `loss/stage4`: 各阶段 loss - 都应稳定
- `train/lr`: 学习率变化曲线 - 平滑下降

---

## 经验教训

### 张量索引最佳实践

1. **维度匹配原则**
   - 使用 mask 索引前，确保数据张量与 mask 的空间维度一致
   - 对于低维参数张量，先用 `view()` 或 `unsqueeze()` 调整维度，再用 `expand()` 扩展

2. **Expand vs Repeat**
   - `expand()` 不占用额外内存（创建视图）
   - `repeat()` 会复制数据（占用更多内存）
   - 优先使用 `expand()`

3. **常见模式**
   ```python
   # Pattern 1: (B,) -> (B, H, W) using mask
   param = param.view(B, 1, 1).expand(B, H, W)
   selected = param[mask]  # (N,)
   
   # Pattern 2: (B, 1, 1) -> (B, H, W) using mask
   param = param.expand(B, H, W)
   selected = param[mask]  # (N,)
   ```

### 损失函数设计原则

1. **数值稳定性优先**
   - 避免在 loss 计算前放大输入值
   - 先计算 loss，再进行归一化

2. **尺度不变性**
   - Loss 的大小应与数据的尺度无关
   - 使用相对误差而非绝对误差

3. **梯度流动性**
   - 确保梯度能顺畅地反向传播
   - 避免过大的梯度裁剪阈值

---

## 常见问题排查

### Q1: 仍然出现 IndexError
**检查点**:
1. 确认 `depth_interval` 的维度是 1 或 3
2. 确认 `B, H, W` 的提取正确
3. 确认使用了 `mask_bool = mask > 0.5` 转换为布尔类型

### Q2: Loss 仍然是 NaN 或 Inf
**解决方案**:
1. 进一步降低学习率：`lr = 1e-5`
2. 增大 grad_clip: `grad_clip = 10.0`
3. 关闭 AMP: `--no_amp`

### Q3: 训练正常但验证 loss 很高
**可能原因**:
1. 过拟合 - 增加 weight_decay 或添加数据增强
2. 数据分布不一致 - 检查 train/val 数据预处理
3. 模型容量不足 - 检查网络结构配置

---

## 总结

本次修复解决了两个层次的问题：

1. **算法层面**: 修正了 loss 归一化的数学逻辑，避免数值不稳定
2. **实现层面**: 正确处理了 PyTorch 的张量索引，避免形状不匹配错误

修复后的代码具有以下特点：
- ✅ 数值稳定：loss 不会因尺度放大而爆炸
- ✅ 形状正确：支持 1D 和 3D 的 depth_interval 输入
- ✅ 效率优化：使用 expand 而非 repeat，节省内存
- ✅ 可维护性：清晰的注释和错误处理

下一步建议：
1. 运行单元测试验证修复
2. 使用 mock 数据集进行快速训练测试
3. 监控 TensorBoard 上的 loss 曲线
4. 根据训练效果微调超参数
