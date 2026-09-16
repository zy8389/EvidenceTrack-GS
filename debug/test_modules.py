# debug/test_modules.py

import os
import sys
import torch
import numpy as np
from easydict import EasyDict

# --- 准备工作：设置路径并导入模块 ---
# 确保可以导入项目中的其他文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene.gaussian_model import GaussianModel
from utils.loss_utils import reprojection_loss
from submodules.GTDCA_attention import GTDCA
from scene.cameras import Camera


def run_tests():
    print("--- 开始模块单元测试 ---")
    device = "cuda"
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，测试将在CPU上运行。")
        device = "cpu"

    # --- 1. 测试 GaussianModel 的新方法 ---
    print("\n[1/3] 正在测试 scene.gaussian_model.py 中的 get_closest_gaussians_for_anchors...")
    try:
        # 修正: 创建一个模拟的args对象来初始化GaussianModel
        mock_args = EasyDict({'sh_degree': 0, 'train_bg': False})  # FSGS的GaussianModel需要args参数
        mock_gaussians = GaussianModel(mock_args)
        mock_gaussians._xyz = torch.randn(10, 3, device=device)

        # 创建3个模拟的3D锚点
        mock_anchors = torch.randn(3, 3, device=device)

        # 调用新方法
        associated_centers = mock_gaussians.get_closest_gaussians_for_anchors(mock_anchors)

        # 验证输出形状
        assert associated_centers.shape == (3, 3)
        print("✅ 测试通过: get_closest_gaussians_for_anchors 方法运行正常，输出形状正确。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # --- 2. 测试 loss_utils.py 中的新损失函数 ---
    print("\n[2/3] 正在测试 utils/loss_utils.py 中的 reprojection_loss...")
    try:
        # 准备模拟数据
        mock_tracks_info = [np.array([[0, 100, 150], [1, 110, 160]], dtype=np.float32)]
        mock_anchors_3d = torch.randn(1, 3, device=device)

        # 修正: 创建更真实的模拟相机，并手动设置必要的属性
        mock_cameras = []
        for i in range(2):
            cam = Camera(colmap_id=i, R=np.identity(3), T=np.zeros(3), FoVx=45, FoVy=45,
                         image=torch.rand(3, 256, 256), gt_alpha_mask=None,
                         image_name=f"img{i}", uid=i, data_device=device)
            cam.full_proj_transform = torch.tensor(cam.world_view_transform.T @ cam.projection_matrix,
                                                   device=device).float()
            mock_cameras.append(cam)

        # 调用损失函数
        loss_val = reprojection_loss(mock_gaussians, mock_cameras, mock_tracks_info, mock_anchors_3d)

        # 验证输出
        assert loss_val.ndim == 0 and loss_val >= 0
        print(f"✅ 测试通过: reprojection_loss 运行正常，返回一个非负标量损失值: {loss_val.item():.4f}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # --- 3. 测试注意力模块 ---
    print("\n[3/3] 正在测试 submodules/GTDCA_attention.py...")
    try:
        mock_gtdca = GTDCA(d_model=64).to(device)

        # 创建模拟输入
        query = torch.randn(1, 10, 64, device=device)  # B, N_g, C
        value = torch.randn(1, 64, 128, 128, device=device)  # B, C, H, W

        # ===================================================================
        # --- 错误修正之处 ---
        # 原来的代码: ref_pts = torch.rand(1, 10, 2, device=device).unsqueeze(2)
        # 这会创建一个形状为 [1, 10, 1, 2] 的4维张量，是错误的。
        # GTDCA模块期望的参考点形状是 [B, N_g, 2]，即一个3维张量。
        # 我们只需移除多余的 .unsqueeze(2) 即可。
        ref_pts = torch.rand(1, 10, 2, device=device)
        # ===================================================================

        traj_pts = torch.rand(1, 5, 2, device=device)

        output = mock_gtdca(query, value, ref_pts, traj_pts)

        # 验证输出形状
        assert output.shape == query.shape
        print("✅ 测试通过: GTDCA 注意力模块实例化和前向传播正常，输出形状正确。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("\n--- 所有单元测试执行完毕 ---")


if __name__ == "__main__":
    run_tests()