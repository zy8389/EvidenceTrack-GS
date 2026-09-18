#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh



def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
           override_color = None, white_bg = False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if min(pc.bg_color.shape) != 0:
        bg_color = torch.tensor([0., 0., 0.]).cuda()

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.get_opacity)
    if confidence.shape != pc.get_opacity.shape:
        raise RuntimeError("Confidence count does not match Gaussian count")
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Task: 修改渲染流程以使用增强的外观特征
    # Task: 确保与现有SH系数的兼容性
    # Requirements: 3.3 - Enhanced rendering with GT-DCA features
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # Check if GT-DCA is available and enabled
        if hasattr(pc, 'is_gt_dca_enabled') and pc.is_gt_dca_enabled():
            # Performance optimization: Only use GT-DCA every 10 iterations to speed up training
            if not hasattr(pc, '_gt_dca_iter_counter'):
                pc._gt_dca_iter_counter = 0
            pc._gt_dca_iter_counter += 1
            
            if pc._gt_dca_iter_counter % 800 == 0:  # Every 800 iterations for T4
                try:
                    # Use GT-DCA enhanced appearance features
                    with torch.no_grad():
                        appearance_features = pc.get_appearance_features(viewpoint_camera, use_gt_dca=True).detach()
                    
                    # Convert GT-DCA features to colors directly
                    if appearance_features.dim() == 2:
                        feature_dim = appearance_features.shape[1]
                        
                        if feature_dim >= 3:
                            # Use first 3 channels as RGB colors
                            colors_precomp = torch.clamp(torch.sigmoid(appearance_features[:, :3]), 0.0, 1.0)
                        elif feature_dim == 1:
                            # Convert grayscale to RGB
                            gray_values = torch.clamp(torch.sigmoid(appearance_features[:, 0:1]), 0.0, 1.0)
                            colors_precomp = gray_values.repeat(1, 3)
                        else:
                            raise ValueError(f"Unsupported GT-DCA feature dimension: {feature_dim}")
                        
                        if pc._gt_dca_iter_counter % 800 == 0:  # Reduce logging frequency
                            print(f"🎨 GT-DCA特征渲染，维度: {appearance_features.shape}")
                        
                    else:
                        # New: Handle 3-D tensors that contain SH coefficients directly
                        # Shape is expected to be (N, coeffs, 3)
                        if appearance_features.dim() == 3 and appearance_features.shape[-1] == 3:
                            if pipe.convert_SHs_python:
                                shs_view = appearance_features.transpose(1, 2).contiguous()  # (N, 3, coeffs)
                                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                            else:
                                shs = appearance_features  # Let rasterizer handle SH->RGB
                            if pc._gt_dca_iter_counter % 100 == 0:
                                print(f"🎨 使用SH特征渲染，形状: {appearance_features.shape}")
                        else:
                            raise ValueError(f"Unsupported GT-DCA feature tensor shape: {appearance_features.shape}")
                    
                except Exception as e:
                    if pc._gt_dca_iter_counter % 100 == 0:  # Reduce error logging frequency
                        print(f"⚠️ GT-DCA失败，回退SH: {e}")
                    # Fallback to standard SH processing
                    if pipe.convert_SHs_python:
                        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                    else:
                        shs = pc.get_features
            else:
                # Use standard SH processing most of the time
                if pipe.convert_SHs_python:
                    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = pc.get_features
        else:
            # Standard SH processing (existing implementation)
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # rendered_image_list, depth_list, alpha_list = [], [], []
    # for i in range(5):
    #     rendered_image, radii, depth, alpha = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp)
    #     rendered_image_list.append(rendered_image)
    #     depth_list.append(depth)
    #     alpha_list.append(alpha)
    # def mean1(t):
    #     return torch.mean(torch.stack(t), 0)
    # rendered_image, depth, alpha = mean1(rendered_image_list), mean1(depth_list), mean1(alpha_list)

    if min(pc.bg_color.shape) != 0:
        rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)  # torch.ones((3, 1, 1)).cuda()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}
