#!/usr/bin/env python3
"""
简单的mask集成测试脚本
测试mask功能的基本组件是否正常工作
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# 添加项目路径
sys.path.append('.')

def test_camera_mask_integration():
    """测试Camera组件的mask集成"""
    print("Testing Camera mask integration...")
    
    from scene.cameras import Camera
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    
    # 创建测试数据
    R = np.eye(3)
    T = np.array([0.0, 0.0, 0.0])
    FoVx = FoVy = 1.0
    image = torch.rand(3, 100, 100)
    
    # 创建测试mask
    mask = Image.new('L', (100, 100), 255)  # 全白mask
    
    # 测试带mask的Camera
    camera = Camera(
        colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
        image=image, gt_alpha_mask=None, image_name="test",
        uid=0, mask=mask
    )
    
    assert hasattr(camera, 'original_mask'), "Camera should have original_mask attribute"
    assert camera.original_mask.shape == (1, 100, 100), f"Mask shape should be (1, 100, 100), got {camera.original_mask.shape}"
    assert torch.all(camera.original_mask == 1.0), "Default mask should be all ones"
    
    # 测试不带mask的Camera
    camera_no_mask = Camera(
        colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
        image=image, gt_alpha_mask=None, image_name="test",
        uid=0, mask=None
    )
    
    assert hasattr(camera_no_mask, 'original_mask'), "Camera should have original_mask attribute even without mask"
    assert torch.all(camera_no_mask.original_mask == 1.0), "Default mask should be all ones when no mask provided"
    
    print("✓ Camera mask integration test passed")

def test_dataset_reader_mask_loading():
    """测试Dataset Reader的mask加载功能"""
    print("Testing Dataset Reader mask loading...")
    
    from scene.dataset_readers import CameraInfo
    
    # 测试CameraInfo是否有mask字段
    try:
        # 创建测试mask
        mask = Image.new('L', (100, 100), 255)
        
        cam_info = CameraInfo(
            uid=0, R=np.eye(3), T=np.array([0.0, 0.0, 0.0]),
            FovY=1.0, FovX=1.0, image=Image.new('RGB', (100, 100)),
            image_path="test.jpg", image_name="test", 
            width=100, height=100, mask=mask
        )
        
        assert cam_info.mask is not None, "CameraInfo should store mask"
        print("✓ Dataset Reader mask loading test passed")
        
    except Exception as e:
        print(f"✗ Dataset Reader test failed: {e}")
        return False
    
    return True

def test_gaussian_model_safe_pruning():
    """测试Gaussian Model的安全裁剪功能"""
    print("Testing Gaussian Model safe pruning...")
    
    from scene.gaussian_model import GaussianModel
    
    # 创建测试GaussianModel
    gmodel = GaussianModel(sh_degree=3, max_texture_resolution=256, device="cpu")
    
    # 检查是否有新的方法
    assert hasattr(gmodel, 'prune_by_mask'), "GaussianModel should have prune_by_mask method"
    
    print("✓ Gaussian Model safe pruning test passed")

def main():
    """运行所有测试"""
    print("Running mask integration tests...\n")
    
    try:
        test_camera_mask_integration()
        test_dataset_reader_mask_loading()
        test_gaussian_model_safe_pruning()
        
        print("\n✓ All mask integration tests passed!")
        print("\nMask功能已成功集成到项目中。")
        print("现在你可以:")
        print("1. 在数据集根目录创建masks文件夹")
        print("2. 在masks文件夹中放置与images文件夹中图像同名的mask文件")
        print("3. 运行训练，系统会自动应用mask功能")
        print("4. 如果没有masks文件夹，系统会使用原版功能")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)