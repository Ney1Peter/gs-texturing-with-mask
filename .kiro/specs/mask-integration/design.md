# Mask Integration Design Document

## Overview

本设计文档描述了如何为3D Gaussian Splatting项目集成mask功能，以在重建过程中去除不需要的对象（如人物）。该功能将在数据加载、初始点云过滤、训练损失计算和模型优化等多个阶段发挥作用，确保被mask标记的区域不参与3D重建过程。

## Architecture

系统采用分层架构，mask功能集成到现有的各个组件中：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset       │    │   Camera        │    │   Training      │
│   Reader        │───▶│   Component     │───▶│   Pipeline      │
│                 │    │                 │    │                 │
│ - Load masks    │    │ - Store mask    │    │ - Apply mask    │
│ - Filter points │    │ - Resize mask   │    │ - Compute loss  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Point Cloud   │    │   Gaussian      │    │   Optimization  │
│   Filter        │    │   Model         │    │   Process       │
│                 │    │                 │    │                 │
│ - 3D projection │    │ - Safe pruning  │    │ - Early pruning │
│ - Mask checking │    │ - State mgmt    │    │ - Loss masking  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. Dataset Reader (scene/dataset_readers.py)

**职责**: 加载mask图像并进行初始点云过滤

**关键修改**:
- 在`readColmapCameras`函数中添加mask加载逻辑
- 在`CameraInfo`类中添加mask字段
- 实现3D点投影检查和过滤逻辑

**接口**:
```python
class CameraInfo(NamedTuple):
    # ... existing fields ...
    mask: Image.Image  # 新增mask字段

def readColmapCameras(...) -> List[CameraInfo]:
    # 加载mask并过滤3D点
```

### 2. Camera Component (scene/cameras.py)

**职责**: 存储和处理mask数据

**关键修改**:
- 在`Camera`类构造函数中添加mask参数
- 实现mask的调整和张量转换
- 存储为`original_mask`属性

**接口**:
```python
class Camera(nn.Module):
    def __init__(self, ..., mask=None, ...):
        # 处理mask数据
        self.original_mask: torch.Tensor
```

### 3. Gaussian Model (scene/gaussian_model.py)

**职责**: 安全地执行点云裁剪操作

**关键修改**:
- 在`_prune_optimizer`中添加优化器存在性检查
- 在`prune_points`中添加安全的张量操作
- 新增`prune_by_mask`方法

**接口**:
```python
class GaussianModel:
    def prune_by_mask(self, valid_mask: torch.Tensor):
        # 安全的mask裁剪
    
    def _prune_optimizer(self, mask):
        # 带安全检查的优化器裁剪
```

### 4. Training Pipeline (train.py)

**职责**: 在训练过程中应用mask

**关键修改**:
- 在损失计算中应用mask
- 在第100步执行早期裁剪
- 确保所有损失函数都考虑mask

**接口**:
```python
def training(...):
    # 应用mask到损失计算
    # 执行早期裁剪
```

### 5. Camera Utils (utils/camera_utils.py)

**职责**: 在相机加载时传递mask数据

**关键修改**:
- 在`loadCam`函数中添加mask参数传递

## Data Models

### Mask Data Flow

```
Dataset Root
├── images/          ──┐
│   ├── img1.jpg       │
│   └── img2.jpg       ├─▶ Dataset Reader ──▶ CameraInfo ──▶ Camera ──▶ Training
└── masks/ (optional) ──┘                     (with mask)    (original_mask)
    ├── img1.jpg
    └── img2.jpg
```

### Mask Processing Pipeline

1. **检测阶段**: 检查与images文件夹同级的masks文件夹是否存在
2. **加载阶段**: 如果masks文件夹存在，从中加载与图像同名的mask文件
3. **验证阶段**: 检查特定mask图像存在性，不存在则创建默认mask
4. **调整阶段**: 将mask调整到渲染分辨率
5. **转换阶段**: 转换为torch张量并移动到GPU
6. **应用阶段**: 在训练和渲染中应用mask

### Point Cloud Filtering Logic

```python
# 伪代码示例
for each 3D_point in colmap_points:
    for each camera_view:
        projected_xy = project_3D_to_2D(3D_point, camera_view)
        for dx, dy in [-1, 0, 1]:  # 检查3x3邻域
            if mask_pixel[projected_xy + (dx, dy)] == 0:  # 黑色区域
                mark_point_as_invalid(3D_point)
                break
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Mask File Discovery and Loading
*For any* dataset with images, the system should correctly identify corresponding mask files in the masks folder and load them with proper fallback to default masks when files are missing.
**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Mask Resolution Consistency  
*For any* loaded mask image, after processing it should have the same spatial dimensions as the corresponding rendered image resolution.
**Validates: Requirements 1.4, 6.2, 6.5**

### Property 3: Point Cloud Filtering Correctness
*For any* 3D point that projects into a mask's black region (including 3x3 neighborhood) in any camera view, that point should be excluded from the initial point cloud.
**Validates: Requirements 2.1, 2.2, 2.3, 2.5**

### Property 4: Loss Computation Masking
*For any* training iteration, all loss computations (L1, SSIM) should only include contributions from pixels where the mask value is greater than 0.5 (white regions).
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 5: Early Pruning Coordinate Transformation
*For any* Gaussian point at iteration 100, the viewspace coordinates should be correctly transformed to image coordinates for mask checking.
**Validates: Requirements 4.2, 4.3**

### Property 6: Mask-Based Point Retention
*For any* Gaussian point during early pruning, points projecting to mask white regions (value > 0.5) should be retained while points projecting to black regions should be removed.
**Validates: Requirements 4.4, 4.5**

### Property 7: Safe Optimizer Handling
*For any* pruning operation, the system should successfully handle both cases where an optimizer exists (training mode) and where it doesn't (inference mode) without errors.
**Validates: Requirements 5.1, 5.2, 5.3**

### Property 8: Tensor Dimension Consistency
*For any* pruning operation, all related tensors (xyz, features, opacity, etc.) should maintain consistent dimensions after the operation.
**Validates: Requirements 5.4, 5.5**

### Property 9: Camera Mask Integration
*For any* camera initialization with mask data, the mask should be properly converted to a torch tensor, moved to the correct device, and stored as original_mask with compatible dimensions.
**Validates: Requirements 6.1, 6.4, 6.5**

### Property 10: Default Mask Creation
*For any* camera initialization without mask data, a default mask of all ones should be created with the same spatial dimensions as the image.
**Validates: Requirements 6.3**

## Error Handling

### Mask File Handling
- **Missing masks folder**: 如果不存在masks文件夹，则禁用mask功能，使用原版功能
- **Missing individual mask files**: 对于存在masks文件夹但缺少特定mask文件的情况，创建默认的全白mask
- **Invalid mask format**: 尝试转换为灰度图像，失败则使用默认mask
- **Size mismatch**: 自动调整mask尺寸以匹配图像分辨率

### Training Safety
- **Optimizer absence**: 在裁剪操作中检查优化器存在性
- **Tensor dimension mismatch**: 验证张量维度兼容性
- **Empty point clouds**: 处理过度裁剪导致的空点云情况

### Memory Management
- **GPU memory**: 及时释放不需要的mask张量
- **Batch processing**: 避免同时加载过多mask数据

## Testing Strategy

### Unit Testing
- 测试mask加载和处理功能
- 测试点云过滤逻辑
- 测试安全裁剪操作
- 测试损失计算中的mask应用

### Property-Based Testing
- 使用随机生成的mask和点云数据验证过滤逻辑
- 测试各种尺寸和格式的mask图像处理
- 验证在不同训练阶段的mask应用效果

### Integration Testing
- 端到端测试完整的mask集成流程
- 测试与现有功能的兼容性
- 验证性能影响在可接受范围内

### Testing Framework
将使用Python的`hypothesis`库进行property-based testing，配置每个测试运行最少100次迭代以确保充分的随机性覆盖。