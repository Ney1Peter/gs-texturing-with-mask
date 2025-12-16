# Mask Integration Requirements Document

## Introduction

为3D Gaussian Splatting项目添加mask功能，用于在重建过程中去除图像中的人物或其他不需要的对象，从而获得更干净的场景重建结果。系统将在数据加载、训练和重建的各个阶段应用mask信息，确保被标记的区域不参与3D重建过程。

## Glossary

- **Mask Image**: 与输入图像对应的二值图像，白色(255)表示保留区域（背景），黑色(0)表示去除区域（人物）
- **Gaussian_Splatting_System**: 基于3D高斯点云的场景重建系统
- **Camera_Component**: 负责处理相机参数和图像数据的组件
- **Dataset_Reader**: 负责加载和预处理训练数据的组件
- **Training_Pipeline**: 执行模型训练的完整流程
- **Point_Cloud_Filter**: 在初始化阶段过滤3D点云的组件
- **Gaussian_Model**: 管理3D高斯点的模型组件

## Requirements

### Requirement 1

**User Story:** 作为一个3D重建用户，我希望能够提供mask图像来指定需要去除的区域，这样重建出的场景就不会包含不需要的对象。

#### Acceptance Criteria

1. WHEN Dataset_Reader加载图像数据 THEN Gaussian_Splatting_System SHALL 在masks文件夹中查找对应的mask图像
2. WHEN mask图像存在 THEN Gaussian_Splatting_System SHALL 加载mask图像并将其传递给Camera_Component
3. WHEN mask图像不存在 THEN Gaussian_Splatting_System SHALL 创建默认的全白mask图像
4. WHEN mask图像被加载 THEN Gaussian_Splatting_System SHALL 将mask图像调整到与渲染分辨率匹配
5. WHEN 处理mask图像 THEN Gaussian_Splatting_System SHALL 将白色区域识别为保留区域，黑色区域识别为去除区域

### Requirement 2

**User Story:** 作为系统开发者，我希望在初始点云加载时就过滤掉落在mask区域的3D点，这样可以从源头减少不需要的点。

#### Acceptance Criteria

1. WHEN 系统读取COLMAP数据 THEN Point_Cloud_Filter SHALL 检查每个3D点在各个视角中的投影位置
2. WHEN 3D点投影到图像坐标 THEN Point_Cloud_Filter SHALL 检查该坐标及其3x3邻域是否落在mask的黑色区域
3. WHEN 3D点在任何视角中落入mask区域 THEN Point_Cloud_Filter SHALL 将该点标记为无效
4. WHEN 点云过滤完成 THEN Point_Cloud_Filter SHALL 报告过滤掉的点数统计信息
5. WHEN 初始化点云 THEN Gaussian_Splatting_System SHALL 只使用未被mask过滤的3D点

### Requirement 3

**User Story:** 作为训练过程的管理者，我希望在训练时只对mask保留区域计算损失，这样模型不会学习被mask的区域。

#### Acceptance Criteria

1. WHEN Training_Pipeline计算渲染损失 THEN Gaussian_Splatting_System SHALL 将渲染图像与mask相乘
2. WHEN Training_Pipeline计算真实图像损失 THEN Gaussian_Splatting_System SHALL 将真实图像与mask相乘
3. WHEN 计算L1损失 THEN Gaussian_Splatting_System SHALL 只在mask保留区域计算损失值
4. WHEN 计算SSIM损失 THEN Gaussian_Splatting_System SHALL 只在mask保留区域计算损失值
5. WHEN 应用alpha_mask THEN Gaussian_Splatting_System SHALL 同时应用alpha_mask和original_mask

### Requirement 4

**User Story:** 作为训练优化的管理者，我希望在训练早期进行基于mask的点云裁剪，这样可以快速移除前景点。

#### Acceptance Criteria

1. WHEN 训练达到第100步 THEN Training_Pipeline SHALL 执行基于mask的点云裁剪
2. WHEN 执行裁剪操作 THEN Gaussian_Splatting_System SHALL 获取当前视角的mask信息
3. WHEN 计算点的投影坐标 THEN Gaussian_Splatting_System SHALL 将viewspace坐标转换为图像坐标
4. WHEN 检查点的有效性 THEN Gaussian_Splatting_System SHALL 保留投影到mask白色区域的点
5. WHEN 裁剪完成 THEN Gaussian_Model SHALL 移除所有投影到mask黑色区域的高斯点

### Requirement 5

**User Story:** 作为模型管理者，我希望Gaussian模型能够安全地处理点云裁剪操作，即使在没有优化器的情况下也能正常工作。

#### Acceptance Criteria

1. WHEN Gaussian_Model执行裁剪操作 THEN Gaussian_Splatting_System SHALL 检查优化器是否存在
2. WHEN 优化器存在 THEN Gaussian_Model SHALL 同时更新模型参数和优化器状态
3. WHEN 优化器不存在 THEN Gaussian_Model SHALL 只更新模型参数而不处理优化器状态
4. WHEN 更新训练相关张量 THEN Gaussian_Model SHALL 检查张量是否已初始化且形状正确
5. WHEN 裁剪操作完成 THEN Gaussian_Model SHALL 确保所有相关张量的维度保持一致

### Requirement 6

**User Story:** 作为相机数据处理者，我希望Camera组件能够正确加载和处理mask数据，并将其与图像数据关联。

#### Acceptance Criteria

1. WHEN Camera_Component初始化 THEN Gaussian_Splatting_System SHALL 接受mask参数并存储为original_mask属性
2. WHEN mask数据存在 THEN Camera_Component SHALL 将mask调整到指定分辨率
3. WHEN mask数据不存在 THEN Camera_Component SHALL 创建全为1的默认mask
4. WHEN 处理mask数据 THEN Camera_Component SHALL 将mask转换为torch张量并移动到指定设备
5. WHEN mask处理完成 THEN Camera_Component SHALL 确保mask与图像具有兼容的维度