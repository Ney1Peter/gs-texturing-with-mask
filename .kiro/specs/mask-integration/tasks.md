# Implementation Plan

- [x] 1. 修改Camera组件以支持mask数据
  - 在Camera类构造函数中添加mask参数处理
  - 实现mask的调整、转换和存储逻辑
  - 确保mask与图像数据的兼容性
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 1.1 为Camera组件编写property测试
  - **Property 9: Camera Mask Integration**
  - **Validates: Requirements 6.1, 6.4, 6.5**

- [ ]* 1.2 为Camera组件编写property测试
  - **Property 10: Default Mask Creation**
  - **Validates: Requirements 6.3**

- [x] 2. 修改Dataset Reader以支持mask加载和点云过滤
  - 在CameraInfo中添加mask字段
  - 实现masks文件夹检测和mask文件加载逻辑
  - 实现3D点投影检查和过滤功能
  - 添加过滤统计信息输出
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 2.1 为mask加载编写property测试
  - **Property 1: Mask File Discovery and Loading**
  - **Validates: Requirements 1.1, 1.2, 1.3**

- [ ]* 2.2 为mask分辨率处理编写property测试
  - **Property 2: Mask Resolution Consistency**
  - **Validates: Requirements 1.4, 6.2, 6.5**

- [ ]* 2.3 为点云过滤编写property测试
  - **Property 3: Point Cloud Filtering Correctness**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 3. 修改Gaussian Model以支持安全的点云裁剪
  - 在_prune_optimizer方法中添加优化器存在性检查
  - 修改prune_points方法以安全处理训练相关张量
  - 添加prune_by_mask方法用于基于mask的裁剪
  - 确保所有裁剪操作的维度一致性
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 3.1 为安全裁剪编写property测试
  - **Property 7: Safe Optimizer Handling**
  - **Validates: Requirements 5.1, 5.2, 5.3**

- [ ]* 3.2 为维度一致性编写property测试
  - **Property 8: Tensor Dimension Consistency**
  - **Validates: Requirements 5.4, 5.5**

- [x] 4. 修改Camera Utils以传递mask数据
  - 在loadCam函数中添加mask参数传递
  - 确保mask数据正确从CameraInfo传递到Camera对象
  - _Requirements: 6.1_

- [x] 5. 修改Training Pipeline以应用mask功能
  - 在损失计算中应用mask到渲染图像和真实图像
  - 实现第100步的早期mask裁剪逻辑
  - 确保所有损失函数都考虑mask区域
  - 添加坐标转换和mask检查逻辑
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 5.1 为损失计算mask应用编写property测试
  - **Property 4: Loss Computation Masking**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [ ]* 5.2 为早期裁剪坐标转换编写property测试
  - **Property 5: Early Pruning Coordinate Transformation**
  - **Validates: Requirements 4.2, 4.3**

- [ ]* 5.3 为mask裁剪逻辑编写property测试
  - **Property 6: Mask-Based Point Retention**
  - **Validates: Requirements 4.4, 4.5**

- [x] 6. 集成测试和验证
  - 使用带有masks文件夹的测试数据集验证完整流程
  - 使用不带masks文件夹的数据集验证原版功能保持不变
  - 验证mask功能对训练性能的影响在可接受范围内
  - 确保与现有功能的完全兼容性
  - _Requirements: All_

- [ ]* 6.1 编写端到端集成测试
  - 测试完整的mask集成流程
  - 验证与现有功能的兼容性
  - 测试性能影响

- [x] 7. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户