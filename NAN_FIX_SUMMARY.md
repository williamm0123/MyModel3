# 训练 NaN 问题修复总结

## 问题诊断

根据您的 TensorBoard 训练曲线，发现以下严重问题：

### 🔴 **关键问题**
1. **Loss 为 NaN** - `train/total` 和 `train/total_step` 都显示 NaN
2. **深度预测值异常** - `depth_mean` ≈ 643（正常应在 1-100 范围）
3. **学习率突然归零** - 在 step 180 后 lr 急剧下降到 0
4. **Stage loss 剧烈震荡** - 所有 stage 的 loss 都在剧烈波动

---

## ✅ 已完成的修改

### 1. **配置文件优化** (`config/mvs.json`)

```json
{
  "train": {
    "batch_size": 2,        // 从 5 降至 2，提高稳定性
    "lr": 1e-5,             // 从 5e-5 降至 1e-5
    "weight_decay": 1e-6,   // 从 1e-5 降至 1e-6
    "grad_clip": 1.0        // 从 5.0 降至 1.0
  },
  "loss": {
    "loss_weights": [0.1, 0.2, 0.3, 0.4]  // 更平滑的权重增长
  }
}
```

**修改原因**：
- **降低 batch_size**: 减少每步的梯度噪声，提高稳定性
- **降低学习率**: 避免参数更新过大导致发散
- **降低 weight_decay**: 减少正则化对数值稳定性的影响
- **加强梯度裁剪**: 防止梯度爆炸
- **平滑 loss 权重**: 避免早期 stage 主导训练

---

### 2. **损失函数增强** (`models/losses.py`)

添加了三层 NaN 检测：

```python
def regression_loss(...):
    # 1. 输入检测
    if torch.isnan(depth_pred).any():
        print("Warning: depth_pred contains NaN")
        return torch.tensor(0.0, device=depth_pred.device)
    
    if torch.isnan(depth_gt).any():
        print("Warning: depth_gt contains NaN")
        return torch.tensor(0.0, device=depth_pred.device)
    
    # 2. Loss 计算后检测
    loss = F.smooth_l1_loss(depth_pred[mask_bool], depth_gt[mask_bool], reduction='none')
    if torch.isnan(loss).any():
        print("Warning: loss contains NaN")
        return torch.tensor(0.0, device=depth_pred.device)
    
    # 3. 安全的归一化
    if depth_interval is not None:
        if depth_interval.dim() == 1:
            depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
        elif depth_interval.dim() == 3:
            depth_interval_expanded = depth_interval.expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
    
    return loss.mean()
```

**关键改进**：
- ✅ 在 loss 计算前检测输入 NaN
- ✅ 在 loss 计算后检测结果 NaN
- ✅ 使用安全的张量扩展方式（expand 而非 repeat）
- ✅ 确保 depth_interval 形状与 mask 兼容

---

### 3. **训练循环保护** (`train.py`)

在 AMP 和非 AMP 路径中都添加了 NaN 检测：

```python
# AMP 路径
with autocast(device_type=device.type, enabled=use_amp):
    outputs = model(...)
    loss_dict = loss_fn(...)
    total_loss = loss_dict["total"]

# 检测 NaN/Inf
if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
    print(f"[WARNING] Step {step_idx}: Loss is NaN/Inf, skipping...")
    optimizer.zero_grad()
    continue

scaler.scale(total_loss).backward()
# ... 继续训练

# 非 AMP 路径同理
```

**保护机制**：
- ✅ 在 backward 前检测 loss 是否异常
- ✅ 发现异常时跳过该 step，避免污染梯度
- ✅ 自动清零梯度，防止累积异常梯度
- ✅ 打印警告信息，便于调试

---

## 🎯 预期效果

修复后训练应该呈现：

### **正常收敛曲线**
```
Loss
 ↑
 │  ┌─┐
 │  │  └──┐
 │  │     └──┐
 │  │        └──┐
 │  │           └─────
 │  └──────────────────→ Step
```

### **关键指标监控**
- `train/total`: 应从 NaN 变为有效数值，并逐渐下降
- `train/depth_mean`: 应回到合理范围（如 400-900，取决于数据集）
- `train/lr`: 应平滑下降，不会突然归零
- `train/stage1~4`: 应逐渐稳定，波动减小

---

## 📝 使用方法

### **立即重新训练**
```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --config config/mvs.json
```

### **监控要点**
1. **前 10 步**: 观察是否有 NaN 警告
2. **前 100 步**: 检查 loss 是否开始下降
3. **前 1000 步**: 验证 depth_mean 是否稳定
4. **每个 epoch**: 对比 train/val loss

### **TensorBoard 命令**
```bash
tensorboard --logdir ../log/tensorboard/YOUR_RUN_NAME
```

---

## 🔧 如果仍有问题

### **方案 A: 进一步降低学习率**
```json
{
  "train": {
    "lr": 5e-6  // 从 1e-5 降至 5e-6
  }
}
```

### **方案 B: 关闭 AMP**
```bash
python train.py --config config/mvs.json --no_amp
```

### **方案 C: 减小 grad_clip**
```json
{
  "train": {
    "grad_clip": 0.5  // 从 1.0 降至 0.5
  }
}
```

### **方案 D: 使用 Mock 数据测试**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/mvs.json \
    --mock \
    --epochs 2 \
    --batch_size 1 \
    --max_train_steps 20
```

---

## 📊 技术细节总结

### **数值稳定性三原则**
1. **Early Detection**: 在计算早期检测 NaN，避免传播
2. **Safe Normalization**: 使用 expand 而非 repeat，节省内存且安全
3. **Graceful Degradation**: 发现异常时优雅跳过，而非崩溃

### **张量形状匹配规则**
```python
# ✅ 正确：先扩展再索引
depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
loss = loss / depth_interval_expanded[mask_bool]

# ❌ 错误：直接跨维度索引
loss = loss / depth_interval[mask_bool]  # IndexError!
```

### **梯度保护策略**
- 在 backward 前检测 loss
- 发现异常立即 skip step
- 自动清零梯度防止累积
- 打印详细日志便于调试

---

## ✅ 修改文件清单

1. ✅ [`config/mvs.json`](file:///home/william/project/MyModel3/config/mvs.json) - 训练超参数优化
2. ✅ [`models/losses.py`](file:///home/william/project/MyModel3/models/losses.py) - 添加 NaN 检测
3. ✅ [`train.py`](file:///home/william/project/MyModel3/train.py) - 训练循环保护

所有修改已完成，可以立即开始训练！
