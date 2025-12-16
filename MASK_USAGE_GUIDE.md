# Mask功能使用指南

## 概述

你的gs-texturing项目现在已经完全支持mask功能和CPU设备支持！🎉

## ✅ 已实现的功能

### 1. Mask功能
- **自动检测**: 系统会自动检测数据集根目录下的`masks`文件夹
- **智能过滤**: 三阶段过滤机制确保人物完全被去除
- **向后兼容**: 没有masks文件夹时自动使用原版功能

### 2. CPU设备支持
- **完全兼容**: 支持`--data_device cpu`参数
- **自动适配**: 所有CUDA硬编码调用已修复

## 🚀 使用方法

### 基本训练（GPU，默认）
```bash
python train.py -s /path/to/your/dataset -r 1
```

### CPU训练
```bash
python train.py -s /path/to/your/dataset --data_device cpu -r 1
```

### 带Mask的训练
1. 在数据集根目录创建`masks`文件夹
2. 放置与图像同名的mask文件（黑白图像，白色=保留，黑色=去除）
3. 正常运行训练命令

## 📁 数据集结构示例

```
your_dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── masks/          # 可选的mask文件夹
│   ├── img1.jpg    # 黑白mask：白色=背景（保留），黑色=人物（去除）
│   ├── img2.jpg
│   └── img3.jpg
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## 🔧 Mask功能工作原理

### 三阶段过滤机制

1. **数据加载时过滤**
   - 在读取COLMAP数据时，检查3D点在各视角的投影
   - 如果3D点在任何视角中落入mask黑色区域（包括3x3邻域），则过滤掉该点
   - 从源头减少不需要的3D点

2. **训练时应用mask**
   - 在计算L1和SSIM损失时，只对mask白色区域计算损失
   - 确保模型不会学习被mask的区域

3. **第100步早期裁剪**
   - 在训练第100步时，执行基于当前视角mask的点云裁剪
   - 快速移除投影到mask黑色区域的高斯点

## 🧪 测试验证

运行以下命令验证功能：

```bash
# 测试mask功能
python test_mask_integration.py

# 测试CPU支持
python test_cpu_support.py
```

## 💡 使用建议

1. **Mask图像格式**: 推荐使用PNG或JPG格式的灰度图像
2. **Mask质量**: 确保mask边缘清晰，避免模糊边界
3. **文件命名**: mask文件名必须与对应的图像文件名完全一致
4. **性能考虑**: 使用mask功能会略微增加处理时间，但能显著提升重建质量

## 🔍 调试信息

训练时会输出以下调试信息：
- 过滤掉的3D点数量统计
- 早期裁剪时保留的高斯点数量
- Mask应用状态

## ⚠️ 注意事项

1. 如果没有`masks`文件夹，系统会自动使用原版功能
2. 如果`masks`文件夹存在但缺少特定mask文件，会创建默认的全白mask
3. CPU模式下训练速度会比GPU慢，但功能完全一致

---

现在你可以开始使用mask功能来获得更干净的3D重建结果了！🎯