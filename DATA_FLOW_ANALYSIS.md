# MyModel3 项目数据流与技术参数详解

## 🌟 项目概述

MyModel3 是一个基于 MVSFormer++ 架构的多视角立体视觉深度估计系统，融合了CNN特征金字塔网络(FPN)和视觉Transformer(DINOv3)的优势，通过侧视注意力(SVA)和特征匹配变换器(FMT)实现高精度的深度估计。

---

## 🔄 完整数据流图解

```
输入数据流:
多视角图像 (B, V, 3, H, W) 
    ↓
[数据预处理]
    ├─ 图像归一化 [0,1]
    ├─ 相机参数读取
    ├─ 深度范围计算
    └─ 投影矩阵构建
    ↓
[特征提取阶段]
    ├─ FPN 编码器 → 多尺度CNN特征
    ├─ DINOv3 编码器 → 视觉Transformer特征
    ├─ SVA 注意力 → 跨视角特征增强
    └─ FPN 解码器 → 融合特征金字塔
    ↓
[FMT 特征匹配]
    ├─ 自注意力处理
    ├─ 交叉注意力匹配
    └─ 增强多视角特征
    ↓
[深度估计阶段]
    ├─ 成本体构建 (4阶段级联)
    ├─ 同源性变换与特征匹配
    ├─ 可见性加权聚合
    └─ 深度回归预测
    ↓
输出:
    ├─ 最终深度图 (B, H, W)
    ├─ 置信度图 (B, H, W)
    └─ 各阶段中间结果
```

---

## 📊 关键技术参数详解

### 1. 输入配置参数

```json
{
  "views": [0, 1, 2],           // 使用的视角索引 (参考视角 + 源视角)
  "batch_size": 2,              // 批次大小
  "image_size": "variable",     // 图像尺寸自适应
  "num_workers": 4              // 数据加载线程数
}
```

### 2. FPN 网络参数

```python
# FPN 配置
feat_chs = [8, 16, 32, 64]     # 各阶段通道数 (stage4→stage1)
                               # 对应分辨率: 1/8, 1/4, 1/2, 1/1

# 编码器输出特征
conv01: (B*V, 8, H, W)         # 最浅层特征
conv11: (B*V, 16, H/2, W/2)    # 第二层特征  
conv21: (B*V, 32, H/4, W/4)    # 第三层特征
conv31: (B*V, 64, H/8, W/8)    # 最深层特征
```

### 3. DINOv3 参数

```python
# DINOv3 配置
arch = "dinov3_vitb16"         # ViT-B/16 架构
patch_size = 16                # 补丁大小
vit_ch = 768                   # 特征维度
pick_layers = [3, 7, 11]       # 提取的Transformer层
freeze = True                  # 冻结预训练权重

# 输出特征
dino_l3: (B*V, 768, Ht, Wt)    # 第3层特征
dino_l7: (B*V, 768, Ht, Wt)    # 第7层特征  
dino_l11: (B*V, 768, Ht, Wt)   # 第11层特征
其中 Ht×Wt = (H/16)×(W/16)     # 特征图分辨率
```

### 4. SVA (侧视注意力) 参数

```python
# SVA 配置
vit_ch = 768                   # 输入特征维度
out_ch = 64                    # 输出特征维度 (匹配FPN stage1)
num_heads = 12                 # 注意力头数
mlp_ratio = 4.0                # MLP扩展比例
cross_interval_layers = 3      # 交叉注意时间隔层数

# 输出特征
sva_out: (B*V, 64, Ht*4, Wt*4) # 上采样后的特征图
```

### 5. FMT (特征匹配变换器) 参数

```python
# FMT 配置
base_channel = 8               # 基础通道数
d_model = 64                   # 模型维度
nhead = 4                      # 注意力头数
layer_names = ["self", "cross", "self", "cross"]  # 层类型序列
mlp_ratio = 4.0                # MLP比率
init_values = 1.0              # 初始化值

# 处理阶段
stage1: (B, V, 64, H/8, W/8)   # 最粗略尺度
stage2: (B, V, 32, H/4, W/4)   # 中等尺度
stage3: (B, V, 16, H/2, W/2)   # 较精细尺度
stage4: (B, V, 8, H, W)        # 最精细尺度
```

---

## 🎯 四阶段深度估计流程

### 阶段1: 粗略估计 (1/8 分辨率)

```python
# 参数配置
ndepths = 32                   # 深度假设数量
base_ch = 64                   # 基础通道数
depth_interval_ratio = 4.0     # 深度间隔比率
temperature = 5.0              # softmax温度

# 深度假设初始化
depth_min = 425.0             # 最小深度 (mm)
depth_max = 935.0             # 最大深度 (mm)
depth_interval = (depth_max - depth_min) / (ndepths - 1)

# 输出
stage1_depth: (B, H/8, W/8)    # 粗略深度图
stage1_conf: (B, H/8, W/8)     # 置信度图
```

### 阶段2: 细化估计 (1/4 分辨率)

```python
# 参数配置
ndepths = 16                   # 减少假设数量
base_ch = 32                   # 降低通道数
depth_interval_ratio = 2.0     # 更精细的间隔
temperature = 5.0              # 保持较高温度

# 深度假设细化
基于stage1预测，在其周围建立更密集的假设分布
```

### 阶段3: 进一步细化 (1/2 分辨率)

```python
# 参数配置
ndepths = 8                    # 继续减少假设
base_ch = 16                   # 更低通道数
depth_interval_ratio = 1.0     # 精细调整
temperature = 5.0              # 适中温度
```

### 阶段4: 最终精炼 (全分辨率)

```python
# 参数配置
ndepths = 4                    # 最少假设数
base_ch = 8                    # 最低通道数
depth_interval_ratio = 0.5     # 最精细调整
temperature = 1.0              # 低温度获得确定性结果

# 最终输出
final_depth: (B, H, W)         # 全分辨率深度图
final_conf: (B, H, W)          # 最终置信度
```

---

## 🧠 核心算法细节

### 1. 同源性变换 (Homography Warping)

```python
# 投影矩阵分解
P = K @ [R|t]                  # 完整投影矩阵
其中:
- K: 内参矩阵 (3×3)
- R: 旋转矩阵 (3×3) 
- t: 平移向量 (3×1)

# 变换公式
x_src = P_src @ D @ P_ref⁻¹ @ x_ref
其中 D = diag([1,1,1/d,1]) 为深度缩放矩阵
```

### 2. 成本体构建

```python
# 组内相关性计算
对于每组g ∈ [1,G]:
cost_vol[g] = mean(Ref_feat[g] × Warped_Src_feat[g])

# 可见性权重计算
visibility_weight = VisNet(cost_volume)

# 加权聚合
final_cost = Σ(cost_vol × visibility_weight) / Σ(visibility_weight)
```

### 3. 深度回归

```python
# 回归方式选择
if depth_type == "regression":
    depth = Σ(probability × depth_values)
elif depth_type == "ce":  # 训练时argmax，推理时回归
    if training:
        depth = depth_values[argmax(probability)]
    else:
        depth = Σ(softmax(prob/T) × depth_values)
```

### 4. 置信度计算

```python
# 不同阶段采用不同策略
if ndepths >= 32:
    confidence = entropy_based_conf(prob_volume, n=4)
elif ndepths >= 16:  
    confidence = entropy_based_conf(prob_volume, n=3)
elif ndepths >= 8:
    confidence = entropy_based_conf(prob_volume, n=2)
else:
    confidence = max(probability)  # 直接最大概率
```

---

## 📈 损失函数设计

### 多阶段加权损失

```python
# 配置参数
loss_weights = [1.0, 1.0, 1.0, 1.0]  # 各阶段权重
depth_types = ["reg", "reg", "reg", "reg"]  # 损失类型

# 总损失计算
total_loss = Σ(weight_i × loss_i)
其中 loss_i 可以为:
- regression_loss: Smooth L1 损失
- cross_entropy_loss: 分类交叉熵损失
```

### 损失归一化

```python
# 深度间隔归一化
normalized_pred = pred / depth_interval
normalized_gt = gt / depth_interval

# 动态截断 (可选)
if clip_func == "dynamic":
    max_loss = depth_range
    loss = clamp(loss, 0, max_loss)
```

---

## ⚡ 性能优化参数

### 训练配置

```python
# 优化器设置
optimizer = AdamW(
    lr = 1e-4,
    weight_decay = 1e-4,
    betas = (0.9, 0.999)
)

# 学习率调度
scheduler = CosineAnnealingLR(
    T_max = epochs × steps_per_epoch,
    eta_min = lr × 0.01
)

# 混合精度训练
use_amp = True
grad_clip = 1.0
```

### 推理优化

```python
# 模型量化 (可选)
quantize_model = False

# 批次推理
batch_inference = True

# 内存优化
torch.backends.cudnn.benchmark = True
```

---

## 📊 数据格式规范

### 输入数据格式

```python
# 图像数据
images: torch.Tensor           # 形状: (B, V, 3, H, W)
dtype: float32                 # 数值范围: [0, 1]

# 相机参数
proj_matrices: Dict[str, Tensor]  # 各阶段投影矩阵
格式: (B, V, 2, 4, 4)          # [0]: extrinsic, [1]: intrinsic

# 深度范围
depth_range: torch.Tensor      # 形状: (B, 2) [min, max]
depth_interval: torch.Tensor   # 形状: (B,) 单位深度间隔
```

### 输出数据格式

```python
# 深度估计结果
outputs = {
    'stage1': {                # 第一阶段输出
        'depth': (B, H/8, W/8),
        'prob_volume': (B, 32, H/8, W/8),
        'depth_values': (B, 32, H/8, W/8)
    },
    'stage2': {...},           # 第二阶段
    'stage3': {...},           # 第三阶段
    'stage4': {...},           # 第四阶段
    'depth': (B, H, W),        # 最终深度
    'photometric_confidence': (B, H, W)  # 置信度
}
```

---

## 🎛️ 可调参数建议

### 初学者推荐设置

```json
{
  "batch_size": 1,              // 小批次避免内存溢出
  "views": [0, 1],              // 减少视角数量
  "depth": {
    "ndepths": [16, 8, 4, 2],   // 减少深度假设
    "base_chs": [32, 16, 8, 4]  // 降低通道数
  }
}
```

### 高性能设置

```json
{
  "batch_size": 4,              // 增大批次提高吞吐量
  "views": [0, 1, 2, 3],        // 增加视角提升精度
  "depth": {
    "ndepths": [64, 32, 16, 8], // 增加深度假设
    "temperatures": [10.0, 10.0, 5.0, 1.0]  // 调整温度参数
  }
}
```

### 内存受限环境

```json
{
  "fpn": {
    "feat_chs": [4, 8, 16, 32]   // 显著降低通道数
  },
  "dinov3": {
    "freeze": true              // 必须冻结以节省显存
  }
}
```

---

## 🔧 调试建议

### 常见问题排查

1. **梯度消失**: 检查学习率是否过小，增加梯度裁剪
2. **内存不足**: 减少批次大小，降低特征通道数
3. **收敛缓慢**: 调整学习率调度策略，检查数据预处理
4. **精度不达标**: 增加深度假设数，调整损失权重

### 监控指标

```python
# 训练监控
- 总损失值趋势
- 各阶段损失分布
- 学习率变化
- 梯度范数
- 深度预测统计 (均值、方差)

# 验证指标  
- 绝对误差 (Abs Err)
- 1mm/2mm/4mm 精度
- 置信度分布
- 深度图可视化质量
```

---

这份文档提供了MyModel3项目的完整技术细节，涵盖了从输入到输出的每个处理步骤和关键参数配置。通过理解这些数据流和参数，您可以更好地进行模型调优和研究工作。