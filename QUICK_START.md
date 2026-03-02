# MyModel3 项目快速入门指南

## 🚀 项目简介

MyModel3 是一个先进的多视角立体视觉(MVS)深度估计系统，基于MVSFormer++架构，融合了CNN特征金字塔网络和视觉Transformer的优势，能够从多张不同角度的照片中重建高质量的3D深度图。

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with ≥ 8GB VRAM (推荐 11GB+)
- **CPU**: 多核处理器 (推荐 8核以上)
- **内存**: ≥ 16GB RAM (推荐 32GB)
- **存储**: ≥ 50GB 可用空间

### 软件依赖
```bash
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.0
```

## 📥 安装部署

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n mymodel3 python=3.9
conda activate mymodel3

# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pillow matplotlib tensorboard scikit-image opencv-python
```

### 2. 数据集准备

```bash
# 下载DTU MVS数据集
wget http://roboimagedata.compute.dtu.dk/?page_id=36
# 或从官方获取: https://roboimagedata.compute.dtu.dk/

# 数据集结构应该如下:
dataset/
├── Rectified/
│   ├── scan1/
│   │   ├── rect_001_0_r5000.png
│   │   ├── rect_002_0_r5000.png
│   │   └── ...
│   └── ...
├── Cameras/
│   ├── 00000000_cam.txt
│   ├── 00000001_cam.txt
│   └── ...
└── Depths/
    ├── scan1/
    │   ├── depth_map_0000.pfm
    │   ├── depth_map_0001.pfm
    │   └── ...
    └── ...
```

### 3. 预训练模型准备

```bash
# 下载DINOv3预训练权重
mkdir -p dataset/pre_trained
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain/dinov3_vitb16_pretrain.pth \
     -O dataset/pre_trained/dinov3_vitb16_pretrain.pth
```

## ⚙️ 配置文件设置

编辑 `config/mvs.json`:

```json
{
  "datapath": "/path/to/your/dataset",           // 数据集根目录
  "train_data_list": "lists/dtu/train.txt",      // 训练扫描列表
  "val_data_list": "lists/dtu/test.txt",         // 验证扫描列表
  
  "train": {
    "epochs": 16,                                // 训练轮数
    "batch_size": 2,                             // 批次大小
    "lr": 1e-4,                                  // 学习率
    "num_workers": 4                             // 数据加载线程数
  },
  
  "dinov3": {
    "weights": "/path/to/dinov3_weights.pth",    // DINOv3权重路径
    "freeze": true                               // 是否冻结预训练权重
  }
}
```

## 🏃‍♂️ 快速运行

### 1. 测试运行 (使用模拟数据)

```bash
# 快速测试模型能否正常运行
python train.py --mock --epochs 1 --batch_size 1

# 查看测试输出
python test.py
```

### 2. 训练模型

```bash
# 基础训练命令
python train.py --config config/mvs.json

# 指定数据路径
python train.py --config config/mvs.json --data_root /path/to/dataset

# 从检查点恢复训练
python train.py --config config/mvs.json --resume runs/checkpoints/latest.pth
```

### 3. 推理测试

```bash
# 运行推理测试
python test.py

# 查看结果
ls runs/vis_*/depth_stages/
```

## 📊 监控训练过程

### TensorBoard 可视化

```bash
# 启动TensorBoard
tensorboard --logdir runs/logs

# 在浏览器中访问: http://localhost:6006
```

监控指标包括：
- 训练/验证损失曲线
- 学习率变化
- 深度预测可视化
- 置信度图

## 🎯 关键参数调优指南

### 初学者友好设置

```json
{
  "train": {
    "batch_size": 1,           // 减少显存占用
    "epochs": 8,              // 减少训练时间
    "lr": 5e-5                // 更保守的学习率
  },
  "views": [0, 1],            // 使用较少视角
  "depth": {
    "ndepths": [16, 8, 4, 2]   // 减少深度假设数
  }
}
```

### 高性能设置

```json
{
  "train": {
    "batch_size": 4,           // 增大批次大小
    "epochs": 32,             // 更长训练时间
    "lr": 2e-4                // 更大学习率
  },
  "views": [0, 1, 2, 3, 4],   // 使用更多视角
  "depth": {
    "ndepths": [64, 32, 16, 8] // 增加深度精度
  }
}
```

## 🔧 常见问题解决

### 1. 内存不足错误

```bash
# 解决方案1: 减少批次大小
--batch_size 1

# 解决方案2: 降低模型复杂度
修改 config/mvs.json:
{
  "fpn": {"feat_chs": [4, 8, 16, 32]},
  "depth": {"ndepths": [16, 8, 4, 2]}
}

# 解决方案3: 启用混合精度
确保配置中 "use_amp": true
```

### 2. 数据加载错误

```bash
# 检查数据路径配置
确认 datapath 指向正确的数据集目录

# 验证数据完整性
python -c "from data.dtu_data import DTUData; dataset = DTUData(cfg)"

# 检查扫描列表文件
cat lists/dtu/train.txt
```

### 3. 训练不收敛

```bash
# 调整学习率
尝试 lr = 5e-5 或 2e-4

# 检查数据预处理
确认图像已正确归一化到 [0,1]

# 增加训练轮数
设置 epochs = 32 或更高
```

## 📈 性能基准

### 预期训练效果

```
Epoch 1-4:   快速收敛，损失显著下降
Epoch 5-12:  稳定优化，精度逐步提升  
Epoch 13-16: 微调阶段，收敛到最优值

典型损失值:
- Stage1 loss: ~0.2-0.5
- Stage4 loss: ~0.05-0.15
- Total loss: ~0.3-0.8
```

### 推理速度

```
输入: 3视角 × 480×640 图像
RTX 3090: ~0.8秒/样本
RTX 4090: ~0.5秒/样本
A100: ~0.3秒/样本
```

## 🧪 开发建议

### 代码调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查中间输出
net = Network(cfg, device)
outputs = net(images, return_intermediate=True)
print(outputs.keys())  # 查看所有中间特征
```

### 自定义数据集

```python
# 继承DTUData类
class CustomDataset(DTUData):
    def __init__(self, custom_config):
        super().__init__(custom_config)
        # 自定义数据加载逻辑
    
    def __getitem__(self, idx):
        # 自定义预处理
        sample = super().__getitem__(idx)
        # 添加自定义处理
        return sample
```

## 📚 进阶学习资源

### 相关论文
- MVSFormer++: 《MVSFormer++: Learning Multi-View Stereo with Adaptive Attention》
- DINOv3: 《DINOv3: Scaling Vision Transformers to 1 Billion Parameters》
- FPN: 《Feature Pyramid Networks for Object Detection》

### 项目文档
- [DATA_FLOW_ANALYSIS.md](DATA_FLOW_ANALYSIS.md) - 详细数据流分析
- [ALGORITHM_DETAILS.md](ALGORITHM_DETAILS.md) - 算法实现细节
- [API_REFERENCE.md](API_REFERENCE.md) - API接口文档

## 🤝 社区支持

如遇到问题，请：
1. 查看FAQ和常见问题解决章节
2. 检查GitHub Issues
3. 提供详细的错误信息和环境配置

---
*Happy coding with MyModel3! 🚀*