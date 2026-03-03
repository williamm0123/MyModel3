# 训练 Loss 增大问题诊断报告

## 问题现象
训练过程中 loss 越来越大，模型无法收敛。

## 根本原因分析

### 1. **损失函数归一化问题** ⚠️ (主要问题)

**位置**: `models/losses.py` - `regression_loss()` 函数

**问题代码**:
```python
# 原代码（有问题）
if depth_interval is not None:
    if depth_interval.dim() == 1:
        depth_interval = depth_interval[:, None, None]
    depth_pred = depth_pred / depth_interval  # ❌ 放大预测值
    depth_gt = depth_gt / depth_interval      # ❌ 放大 GT 值
```

**问题分析**:
- DTU 数据集中 `depth_interval ≈ 2.5` (由 `(905-425)/192` 计算得到)
- 当除以这个小于 1 的值时，会将 depth_pred 和 depth_gt 放大约 2.5 倍
- Smooth L1 loss 对异常值敏感，放大的值会导致 loss 激增
- 例如：原始误差为 10，放大后变成 25，loss 从 10 增加到 25

**修复方案**:
```
# 修复后的代码
loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], reduction='none')

# 在计算 loss 后再进行归一化
if depth_interval is not None:
    if depth_interval.dim() == 1:
        depth_interval = depth_interval[:, None, None]
    loss = loss / depth_interval[mask]  # ✅ 归一化 loss，而不是放大输入
```

---

### 2. **超参数配置不合理** ⚠️

**位置**: `config/mvs.json`

#### 2.1 学习率过高
- **原配置**: `"lr": 1e-4`
- **新配置**: `"lr": 5e-5`
- **原因**: AdamW 优化器通常使用较小的学习率，1e-4 可能导致梯度更新过大

#### 2.2 权重衰减过大
- **原配置**: `"weight_decay": 1e-4`
- **新配置**: `"weight_decay": 1e-5`
- **原因**: 过大的 weight decay 会惩罚所有参数，包括 LayerNorm 等不应正则化的参数

#### 2.3 梯度裁剪过小
- **原配置**: `"grad_clip": 1.0`
- **新配置**: `"grad_clip": 5.0`
- **原因**: 1.0 的阈值太小，会过度裁剪梯度，导致有效梯度信息丢失

#### 2.4 Loss 权重配置不当
- **原配置**: `"loss_weights": [1.0, 1.0, 1.0, 1.0]`
- **新配置**: `"loss_weights": [0.25, 0.5, 0.75, 1.0]`
- **原因**: 
  - Stage1 (1/8 尺度) 特征粗糙，预测不准确，应给予较小权重
  - Stage4 (全分辨率) 最精细，应给予最大权重
  - 渐进式权重有助于稳定训练

---

### 3. **学习率调度器问题** ⚠️

**位置**: `train.py` - CosineAnnealingLR 配置

**原代码**:
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, args.epochs * max(1, steps_per_epoch)),
    eta_min=lr * 0.01,  # ❌ 最小学习率过低
)
```

**问题**:
- `eta_min=lr*0.01` 会使学习率快速下降到接近 0
- 导致训练后期几乎不更新参数

**修复**:
```python
eta_min=lr * 0.1  # ✅ 保持一定的学习能力
```

---

## 修复方案总结

### ✅ 已修复的问题

1. **`models/losses.py`** - 修改了 `regression_loss()` 函数的归一化逻辑
2. **`config/mvs.json`** - 调整了训练超参数：
   - lr: 1e-4 → 5e-5
   - weight_decay: 1e-4 → 1e-5
   - grad_clip: 1.0 → 5.0
   - loss_weights: [1.0,1.0,1.0,1.0] → [0.25,0.5,0.75,1.0]
3. **`train.py`** - 调整了 CosineAnnealingLR 的 eta_min: lr*0.01 → lr*0.1

---

## 预期效果

修复后，训练应该呈现以下特征：

1. **初始阶段** (Epoch 1-4):
   - Loss 快速下降但保持稳定
   - 不会出现爆炸性增长
   - Stage4 的 loss 主导，Stage1 辅助

2. **中期阶段** (Epoch 5-10):
   - Loss 平稳下降
   - Train/Val loss 差距逐渐缩小
   - 各 stage 的 loss 权重逐渐平衡

3. **后期阶段** (Epoch 11-16):
   - Loss 收敛到稳定值
   - 验证集 loss 与训练集 loss 接近
   - 模型达到最优状态

---

## 验证方法

### 1. 检查 TensorBoard 日志
```bash
tensorboard --logdir ../log/tensorboard/YOUR_RUN_NAME
```

观察指标：
- `loss/train_total_step`: 每个 step 的总 loss
- `loss/train_total_epoch`: 每个 epoch 的平均 loss
- `train/lr`: 学习率变化曲线
- `loss/stage1`, `loss/stage2`, `loss/stage3`, `loss/stage4`: 各阶段 loss

### 2. 预期曲线特征

**正常训练**:
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

**异常训练**(当前问题):
```
Loss
 ↑     ╱│
 │    ╱ │
 │   ╱  │
 │  ╱   │
 │ ╱    │
 │╱     │
 └──────┴─────────────→ Epoch
```

### 3. 检查点文件
查看保存的 checkpoint：
```bash
ls -lh ../log/checkpoints/YOUR_RUN_NAME/
```

关注：
- `latest.pth`: 最新模型
- `best.pth`: 最优模型（metric 最小）

---

## 其他建议

### 1. 使用 Warmup 策略
如果初始 loss 仍然不稳定，可以添加学习率 warmup：

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=args.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% 的步数用于 warmup
    anneal_strategy='cos',
)
```

### 2. 梯度累积
如果显存允许，可以增加 batch size 或使用梯度累积：

```python
# 在 train_one_epoch() 中
accumulation_steps = 2  # 每 2 步更新一次
for step_idx, batch in enumerate(loader):
    # ... 前向传播 ...
    
    if scaler is not None:
        scaler.scale(total_loss).backward()
        
        if (step_idx + 1) % accumulation_steps == 0:
            # 执行 optimizer.step()
    else:
        total_loss.backward()
        
        if (step_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### 3. 混合精度训练稳定性
如果使用 AMP 训练不稳定，可以尝试：

```python
# 在 GradScaler 初始化时设置 init_scale
scaler = GradScaler(
    device=device.type,
    enabled=(use_amp and device.type == "cuda"),
    init_scale=65536.0,  # 默认值，可以调小如 1024.0
)
```

### 4. 数据增强
如果过拟合严重，可以添加数据增强：

```python
# 在 data/dtu_data.py 的 __getitem__ 中添加
if self.split == "train":
    # 随机颜色抖动
    images = torch_jitter(images, brightness=0.1, contrast=0.1)
    # 随机翻转
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-1])
        depth_gt = torch.flip(depth_gt, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
```

---

## 快速测试

使用 mock 数据集快速验证修复效果：

```bash
# 单卡调试模式
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/mvs.json \
    --mock \
    --epochs 2 \
    --batch_size 2 \
    --max_train_steps 10

# 观察输出中的 loss 变化
# 应该看到 loss 逐渐下降或保持稳定
```

---

## 常见问题排查

### Q1: Loss 仍然是 NaN 或 Inf
**原因**: 梯度爆炸或数值不稳定

**解决方案**:
1. 进一步降低学习率：`lr = 1e-5`
2. 增大 grad_clip: `grad_clip = 10.0`
3. 关闭 AMP: `--no_amp`

### Q2: Loss 下降很慢
**原因**: 学习率过小或模型容量不足

**解决方案**:
1. 检查 DINOv3 权重是否正确加载
2. 确认 FPN 通道数是否匹配
3. 尝试增加 lr: `lr = 1e-4`

### Q3: Train loss 下降但 Val loss 上升
**原因**: 过拟合

**解决方案**:
1. 增加 weight_decay: `1e-4`
2. 添加数据增强
3. 早停策略（early stopping）

---

## 总结

训练 loss 增大的主要原因是：
1. **损失函数归一化错误** (最主要)
2. **超参数配置过于激进**
3. **学习率调度器设置不当**

修复这些问题后，训练应该能够正常收敛。如果仍有问题，请检查：
- 数据加载是否正确
- 投影矩阵是否正确
- 深度范围是否合理
- 模型权重初始化是否正常
