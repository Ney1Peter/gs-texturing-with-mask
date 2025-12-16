#!/usr/bin/env python3
"""
测试CPU设备支持
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.append('.')

def test_cpu_device_support():
    """测试CPU设备支持"""
    print("Testing CPU device support...")
    
    try:
        from scene.cameras import Camera
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud
        
        # 测试Camera类的CPU支持
        print("Testing Camera with CPU device...")
        R = np.eye(3)
        T = np.array([0.0, 0.0, 0.0])
        FoVx = FoVy = 1.0
        image = torch.rand(3, 100, 100)
        mask = Image.new('L', (100, 100), 255)
        
        camera = Camera(
            colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            image=image, gt_alpha_mask=None, image_name="test",
            uid=0, mask=mask, data_device="cpu"
        )
        
        assert camera.data_device == torch.device("cpu"), f"Expected CPU device, got {camera.data_device}"
        assert camera.original_image.device == torch.device("cpu"), f"Image should be on CPU, got {camera.original_image.device}"
        assert camera.original_mask.device == torch.device("cpu"), f"Mask should be on CPU, got {camera.original_mask.device}"
        print("✓ Camera CPU support test passed")
        
        # 测试GaussianModel的CPU支持
        print("Testing GaussianModel with CPU device...")
        gmodel = GaussianModel(sh_degree=3, max_texture_resolution=256, device="cpu")
        assert gmodel.device == torch.device("cpu"), f"Expected CPU device, got {gmodel.device}"
        
        # 测试create_from_pcd
        points = np.random.random((100, 3))
        colors = np.random.random((100, 3))
        normals = np.zeros((100, 3))
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        gmodel.create_from_pcd(pcd, num_samples=50)
        
        assert gmodel._xyz.device == torch.device("cpu"), f"XYZ should be on CPU, got {gmodel._xyz.device}"
        assert gmodel._features_dc.device == torch.device("cpu"), f"Features DC should be on CPU, got {gmodel._features_dc.device}"
        assert gmodel._opacity.device == torch.device("cpu"), f"Opacity should be on CPU, got {gmodel._opacity.device}"
        
        print("✓ GaussianModel CPU support test passed")
        
        print("\n✓ All CPU device support tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ CPU device support test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpu_device_support()
    sys.exit(0 if success else 1)