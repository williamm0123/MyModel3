# MyModel3 核心算法实现详解

## 🎯 数学基础与理论框架

### 1. 多视角立体视觉基础

#### 几何约束关系

在多视角立体视觉中，空间点 P(X,Y,Z) 在不同视角下的成像满足以下几何关系：

```
λ₁ * m₁ = K₁ * [R₁|t₁] * P
λ₂ * m₂ = K₂ * [R₂|t₂] * P
...
λₙ * mₙ = Kₙ * [Rₙ|tₙ] * P
```

其中：
- mᵢ: 第i视角的像素坐标 (3×1齐次坐标)
- Kᵢ: 第i视角的内参矩阵 (3×3)
- [Rᵢ|tᵢ]: 第i视角的外参矩阵 (3×4)
- λᵢ: 深度因子 (标量)
- P: 世界坐标系下的3D点 (4×1齐次坐标)

#### 重投影误差最小化

目标是最小化所有视角的重投影误差：

```
E(P) = Σᵢ ||mᵢ - π(Kᵢ[Rᵢ|tᵢ]P)||²
```

其中 π(·) 表示齐次坐标的透视除法操作。

### 2. 成本体构建理论

#### 特征相似性度量

对于参考视角特征 Fᵣᵉᶠ 和源视角特征 Fₛᵣᶜ，定义相似性度量：

```
similarity(Fᵣᵉᶠ, Fₛᵣᶜ) = corr(Fᵣᵉᶠ, Fₛᵣᶜ) = (Fᵣᵉᶠ ⊙ Fₛᵣᶜ) / (||Fᵣᵉᶠ|| × ||Fₛᵣᶜ||)
```

其中 ⊙ 表示逐元素乘积，||·|| 表示L2范数。

#### 深度假设采样

在深度范围 [dₘᵢₙ, dₘₐₓ] 内均匀采样D个假设：

```
dⱼ = dₘᵢₙ + j × (dₘₐₓ - dₘᵢₙ)/(D-1),  j = 0,1,...,D-1
```

### 3. 逆深度参数化

为了更好地处理远近场景，采用逆深度参数化：

```
inverse_depth = 1/depth
```

这样深度假设变为：

```
δⱼ = 1/dₘₐₓ + j × (1/dₘᵢₙ - 1/dₘₐₓ)/(D-1)
```

优势：
- 远距离区域采样更密集
- 近距离区域采样更稀疏
- 符合实际场景深度分布特性

---

## 🧠 网络架构详细解析

### 1. FPN (Feature Pyramid Network) 实现

#### 编码器结构

```python
class FPNEncoder(nn.Module):
    def __init__(self):
        # Stage 1: 最浅层特征提取
        self.conv0 = nn.Sequential(
            Conv2d(3, 8, 3, 2, 1),    # 1/2 下采样
            Conv2d(8, 8, 3, 1, 1)
        )
        
        # Stage 2: 
        self.conv1 = nn.Sequential(
            Conv2d(8, 16, 3, 2, 1),   # 1/4 下采样
            Conv2d(16, 16, 3, 1, 1)
        )
        
        # Stage 3:
        self.conv2 = nn.Sequential(
            Conv2d(16, 32, 3, 2, 1),  # 1/8 下采样
            Conv2d(32, 32, 3, 1, 1)
        )
        
        # Stage 4: 最深层
        self.conv3 = nn.Sequential(
            Conv2d(32, 64, 3, 2, 1),  # 1/16 下采样
            Conv2d(64, 64, 3, 1, 1)
        )
```

#### 解码器结构（带跳跃连接）

```python
class FPNDecoder(nn.Module):
    def forward(self, conv0, conv1, conv2, conv3):
        # 自底向上路径
        feat4 = self.lat_layer4(conv3)  # 1/16
        
        # 上采样 + 跳跃连接
        feat3 = self.lat_layer3(conv2) + F.interpolate(
            feat4, scale_factor=2, mode='bilinear'
        )  # 1/8
        
        feat2 = self.lat_layer2(conv1) + F.interpolate(
            feat3, scale_factor=2, mode='bilinear'
        )  # 1/4
        
        feat1 = self.lat_layer1(conv0) + F.interpolate(
            feat2, scale_factor=2, mode='bilinear'
        )  # 1/2
        
        # 最终上采样到原始分辨率
        feat0 = F.interpolate(feat1, scale_factor=2, mode='bilinear')  # 1/1
        
        return feat1, feat2, feat3, feat4
```

### 2. DINOv3 特征提取

#### Transformer 编码器块

```python
class DINOv3Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU
        )
    
    def forward(self, x):
        # 注意力残差连接
        x = x + self.attn(self.norm1(x))
        # MLP残差连接
        x = x + self.mlp(self.norm2(x))
        return x
```

#### 多尺度特征提取

```python
def extract_multiscale_features(self, x):
    features = []
    
    # 通过patch embedding
    x = self.patch_embed(x)  # (B, N, dim)
    
    # 添加位置编码
    x = x + self.pos_embed
    
    # 逐层提取特征
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        
        # 在指定层提取特征
        if i in self.pick_layers:
            # 重塑为特征图
            feat = x[:, 1:].transpose(1, 2)  # 移除cls token
            feat = feat.view(B, dim, H, W)
            features.append(feat)
    
    return features
```

### 3. SVA (Side View Attention) 实现

#### 交叉视角注意力机制

```python
class SideViewAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, query_views, key_value_views):
        """
        query_views: (B, V_q, N, C) - 查询视角
        key_value_views: (B, V_kv, N, C) - 键值视角
        """
        B, V_q, N, C = query_views.shape
        V_kv = key_value_views.shape[1]
        
        # 投影
        q = self.q_proj(query_views).reshape(B, V_q, N, self.num_heads, C // self.num_heads)
        kv = self.kv_proj(key_value_views).reshape(B, V_kv, N, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(dim=3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, V_q, N, heads, V_kv*N)
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        out = (attn @ v).reshape(B, V_q, N, C)
        return self.out_proj(out)
```

### 4. FMT (Feature Matching Transformer) 实现

#### 自注意力模块

```python
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, feat):
        # feat: (B*V, C, H, W)
        B_V, C, H, W = feat.shape
        feat_flat = feat.view(B_V, C, H*W).permute(0, 2, 1)  # (B*V, H*W, C)
        
        # 自注意力
        attn_out, _ = self.self_attn(feat_flat, feat_flat, feat_flat)
        
        # 残差连接
        out = self.norm(feat_flat + attn_out)
        return out.permute(0, 2, 1).view(B_V, C, H, W)
```

#### 交叉注意力模块

```python
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query_feat, support_feats):
        """
        query_feat: (B, C, H, W) - 查询特征
        support_feats: (B, V-1, C, H, W) - 支持特征
        """
        B, C, H, W = query_feat.shape
        V_minus_1 = support_feats.shape[1]
        
        # 展平
        query_flat = query_feat.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        support_flat = support_feats.view(B, V_minus_1*C, H*W).permute(0, 2, 1)  # (B, H*W, (V-1)*C)
        
        # 交叉注意力
        attn_out, _ = self.cross_attn(query_flat, support_flat, support_flat)
        
        # 残差连接
        out = self.norm(query_flat + attn_out)
        return out.permute(0, 2, 1).view(B, C, H, W)
```

---

## 🎯 深度估计核心算法

### 1. 成本体构建算法

#### 同源性变换实现

```python
def homo_warping_3d(src_feat, src_proj, ref_proj, depth_values):
    """
    Args:
        src_feat: (B, C, H, W) - 源视角特征
        src_proj: (B, 4, 4) - 源视角投影矩阵
        ref_proj: (B, 4, 4) - 参考视角投影矩阵  
        depth_values: (B, D, H, W) - 深度假设
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    
    # 构建像素坐标网格
    y, x = torch.meshgrid(
        torch.arange(H, device=src_feat.device),
        torch.arange(W, device=src_feat.device),
        indexing='ij'
    )
    y, x = y.contiguous(), x.contiguous()
    coord = torch.stack([x, y, torch.ones_like(x)], dim=0)  # (3, H, W)
    coord = coord.unsqueeze(0).repeat(B, 1, 1, 1).view(B, 3, -1)  # (B, 3, H*W)
    coord = torch.cat([
        coord,
        torch.ones(B, 1, H*W, device=coord.device)
    ], dim=1)  # (B, 4, H*W)
    
    # 计算变换矩阵
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # (B, 4, 4)
    
    # 对每个深度假设进行变换
    warped_feats = []
    for d in range(D):
        # 构建深度缩放矩阵
        depth_scale = depth_values[:, d:d+1, :, :].view(B, 1, H*W)  # (B, 1, H*W)
        depth_diag = torch.diag_embed(
            torch.cat([torch.ones(B, 2, H*W), 1.0/depth_scale, torch.ones(B, 1, H*W)], dim=1)
        )  # (B, 4, 4, H*W)
        
        # 应用变换
        proj_coord = torch.matmul(
            proj.unsqueeze(-1), 
            depth_diag
        ).sum(dim=2)  # (B, 4, H*W)
        proj_coord = proj_coord[:, :3] / (proj_coord[:, 3:4] + 1e-8)  # 透视除法
        
        # 双线性插值采样
        grid = proj_coord[:, :2].view(B, 2, H, W).permute(0, 2, 3, 1)  # (B, H, W, 2)
        grid[..., 0] = grid[..., 0] / (W-1) * 2 - 1  # 归一化到[-1,1]
        grid[..., 1] = grid[..., 1] / (H-1) * 2 - 1
        
        warped_feat = F.grid_sample(
            src_feat, grid, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )  # (B, C, H, W)
        
        warped_feats.append(warped_feat)
    
    return torch.stack(warped_feats, dim=2)  # (B, C, D, H, W)
```

### 2. 组内相关性计算

```python
def groupwise_correlation(ref_feat, warped_feat, groups):
    """
    Args:
        ref_feat: (B, C, H, W) - 参考视角特征
        warped_feat: (B, C, D, H, W) - 变换后的源视角特征
        groups: 分组数
    """
    B, C, D, H, W = warped_feat.shape
    
    if groups < C:
        # 分组相关性
        ref_groups = ref_feat.view(B, groups, C//groups, 1, H, W)
        warped_groups = warped_feat.view(B, groups, C//groups, D, H, W)
        correlation = (ref_groups * warped_groups).mean(dim=2)  # (B, G, D, H, W)
    else:
        # 直接相关性
        ref_expanded = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
        correlation = ref_expanded * warped_feat  # (B, C, D, H, W)
    
    return correlation
```

### 3. 可见性权重计算

```python
class VisibilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 16, 3, 1, 1)  # 假设32组相关性
        self.conv2 = nn.Conv2d(16, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, cost_volume):
        """
        cost_volume: (B, G, D, H, W) - 成本体
        """
        # 聚合深度维度
        cost_agg = cost_volume.mean(dim=2)  # (B, G, H, W)
        
        # 网络处理
        x = F.relu(self.conv1(cost_agg))
        x = F.relu(self.conv2(x))
        visibility = self.sigmoid(self.conv3(x))  # (B, 1, H, W)
        
        return visibility
```

### 4. 成本正则化网络

```python
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # 3D卷积编码器
        self.conv1 = Conv3d(in_channels, base_channels*2, stride=2)
        self.conv2 = Conv3d(base_channels*2, base_channels*2)
        self.conv3 = Conv3d(base_channels*2, base_channels*4, stride=2) 
        self.conv4 = Conv3d(base_channels*4, base_channels*4)
        self.conv5 = Conv3d(base_channels*4, base_channels*8, stride=2)
        self.conv6 = Conv3d(base_channels*8, base_channels*8)
        
        # 3D转置卷积解码器
        self.deconv1 = Deconv3d(base_channels*8, base_channels*4, stride=2)
        self.deconv2 = Deconv3d(base_channels*4, base_channels*2, stride=2) 
        self.deconv3 = Deconv3d(base_channels*2, base_channels, stride=2)
        
        # 输出层
        self.prob = nn.Conv3d(base_channels, 1, 3, 1, 1)
    
    def forward(self, x):
        # 编码
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        
        # 解码 + 跳跃连接
        x = conv4 + self.deconv1(x)
        x = conv2 + self.deconv2(x) 
        x = conv0 + self.deconv3(x)
        
        prob = self.prob(x)
        return prob
```

### 5. 深度回归算法

```python
def depth_regression(prob_volume, depth_values):
    """
    Args:
        prob_volume: (B, D, H, W) - 概率分布
        depth_values: (B, D, H, W) - 深度假设值
    """
    # 加权平均
    depth = torch.sum(prob_volume * depth_values, dim=1)
    return depth

def conf_regression(prob_volume, n=4):
    """
    基于熵的置信度计算
    """
    # 取前n个最高概率
    top_probs, _ = torch.topk(prob_volume, n, dim=1)
    
    # 计算熵作为不确定性度量
    entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-8), dim=1)
    
    # 转换为置信度 (低熵 = 高置信度)
    confidence = torch.exp(-entropy)
    return confidence
```

---

## 📊 级联细化策略

### 深度假设调度算法

```python
def schedule_inverse_range(prev_depth, prev_depth_values, ndepths, ratio, H, W):
    """
    基于前一阶段预测细化深度假设
    
    Args:
        prev_depth: (B, H, W) - 前一阶段深度预测
        prev_depth_values: (B, D_prev, H, W) - 前一阶段深度假设
        ndepths: 新的假设数量
        ratio: 区间缩小比率
        H, W: 目标分辨率
    """
    B = prev_depth.shape[0]
    
    # 计算前一阶段的深度间隔
    depth_interval = 1.0/prev_depth_values[:, 2] - 1.0/prev_depth_values[:, 1]
    
    # 新的深度范围围绕预测值
    inv_min_depth = 1.0/prev_depth + ratio * depth_interval
    inv_max_depth = 1.0/prev_depth - ratio * depth_interval
    
    # 防止负深度
    inv_max_depth = torch.clamp(inv_max_depth, min=0.002)
    
    # 在新范围内均匀采样
    indices = torch.arange(0, ndepths, device=prev_depth.device, dtype=prev_depth.dtype)
    indices = indices.view(1, -1, 1, 1).expand(B, ndepths, H//2, W//2)
    
    # 插值到目标分辨率
    inv_depth_hypo = inv_max_depth.unsqueeze(1)[:, :, ::2, ::2] + \
                    (inv_min_depth - inv_max_depth).unsqueeze(1)[:, :, ::2, ::2] * \
                    (indices / (ndepths - 1))
    
    inv_depth_hypo = F.interpolate(
        inv_depth_hypo.unsqueeze(1),
        size=[ndepths, H, W],
        mode='trilinear',
        align_corners=True
    ).squeeze(1)
    
    return 1.0 / inv_depth_hypo
```

---

## 🎯 损失函数数学表达

### 1. 回归损失

```python
def regression_loss(depth_pred, depth_gt, mask, depth_interval=None):
    """
    Smooth L1 损失
    """
    # 有效像素掩码
    valid_mask = mask > 0.5
    
    # 深度间隔归一化
    if depth_interval is not None:
        if depth_interval.dim() == 1:
            depth_interval = depth_interval[:, None, None]
        depth_pred = depth_pred / depth_interval
        depth_gt = depth_gt / depth_interval
    
    # Smooth L1 损失计算
    diff = torch.abs(depth_pred[valid_mask] - depth_gt[valid_mask])
    loss = torch.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    
    return loss.mean()
```

### 2. 分类交叉熵损失

```python
def cross_entropy_loss(prob_volume_pre, depth_values, depth_gt, mask, inverse_depth=True):
    """
    深度分类损失
    """
    B, D, H, W = depth_values.shape
    valid_mask = mask > 0.5
    
    # 处理逆深度顺序
    if inverse_depth:
        depth_values = torch.flip(depth_values, dims=[1])
        prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
    
    # 找到GT对应的bin索引
    depth_gt_expanded = depth_gt.unsqueeze(1).expand(-1, D, -1, -1)
    
    # 计算每个bin的边界
    intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2
    intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)
    
    depth_values_right = depth_values + intervals
    
    # 确定GT落在哪个bin
    gt_bin_indices = (depth_values_right <= depth_gt_expanded).sum(dim=1) - 1
    gt_bin_indices = torch.clamp(gt_bin_indices, 0, D-1)
    
    # 应用掩码并计算CE损失
    valid_pixels = valid_mask.bool()
    gt_labels = gt_bin_indices[valid_pixels]  # (N,)
    logits = prob_volume_pre.permute(0, 2, 3, 1)[valid_pixels, :]  # (N, D)
    
    return F.cross_entropy(logits, gt_labels, reduction='mean')
```

### 3. 多阶段加权损失

```python
def multiscale_loss(outputs, gt_depths, masks, weights):
    """
    多阶段损失加权求和
    """
    total_loss = 0.0
    stage_losses = {}
    
    for i, stage_key in enumerate(['stage1', 'stage2', 'stage3', 'stage4']):
        if stage_key in outputs:
            stage_pred = outputs[stage_key]['depth']
            stage_gt = gt_depths[stage_key]
            stage_mask = masks[stage_key]
            stage_weight = weights[i]
            
            # 计算该阶段损失
            stage_loss = regression_loss(stage_pred, stage_gt, stage_mask)
            
            stage_losses[stage_key] = stage_weight * stage_loss
            total_loss += stage_losses[stage_key]
    
    stage_losses['total'] = total_loss
    return stage_losses
```

---

## 📈 性能优化技术

### 1. 混合精度训练

```python
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = model
        self.optimizer = optimizer
    
    def step(self, loss):
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 缩放损失并反向传播
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

### 2. 内存优化策略

```python
def memory_efficient_forward(model, inputs):
    """
    内存高效的前向传播
    """
    with torch.cuda.amp.autocast():
        # 分阶段计算避免同时存储所有中间特征
        features = model.feature_extractor(inputs)
        
        # 逐步处理每个阶段
        stage_outputs = {}
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage_feature = features[stage]
            with torch.no_grad():  # 减少梯度存储
                stage_result = model.depth_estimator(stage_feature)
            stage_outputs[stage] = stage_result
    
    return stage_outputs
```

---

这份文档详细解释了MyModel3项目的核心算法实现，包括数学原理、网络架构、关键算法步骤和优化技术。通过理解这些实现细节，您可以更好地进行算法改进和性能调优。