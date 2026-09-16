#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
                                            #'{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)



def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = copy.deepcopy(views[0])

    if source_path.find('llff') != -1:
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    elif source_path.find('360') != -1:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)
    # final_video = cv2.VideoWriter(os.path.join('/ssd1/zehao/gs_release/video/', str(iteration), model_path.split('/')[-1] + '.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(args)
        # 如果启用了 GT-DCA，则在渲染阶段显式启用（确保使用增强外观特征）
        if getattr(args, 'use_gt_dca', False):
            try:
                gaussians.enable_gt_dca()
            except Exception as e:
                print(f"[GT-DCA] 渲染阶段启用失败，将继续使用标准 SH 特征: {e}")
        # 关键修复：总是将完整的 'args' 对象传递给 Scene 类
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 如果是视频模式，调用 render_video
    if args.video:
        video_cameras = scene.getTestCameras()
        if not video_cameras:
            print("Warning: No test cameras found. Using training cameras for video generation setup.")
            video_cameras = scene.getTrainCameras()
        
        if video_cameras:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, video_cameras,
                         gaussians, pipeline, background, args.fps)
        else:
            print("Error: No cameras (neither test nor train) available to generate video.")
        return # 生成完视频后结束

    # 如果是图片渲染模式 (默认)
    if not args.skip_train:
        # 为了评估，我们通常只关心测试集，可以暂时跳过训练集的渲染
        # print("Rendering training set...")
        # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        pass

    if not args.skip_test:
        print("Rendering test set...")
        # 检查测试集是否为空
        if scene.getTestCameras():
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
        else:
            print("Warning: No test cameras found to render.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--data_type", type=str, default="colmap", help="Type of dataset, e.g., 'colmap', 'blender', '360'")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true")
    
    # 混合精度渲染选项
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision rendering")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="AMP data type")
    
    # Task: 添加GT-DCA启用/禁用的配置选项 - 渲染脚本支持
    # GT-DCA: Add arguments for GT-DCA appearance modeling in rendering
    parser.add_argument("--use_gt_dca", action="store_true", help="Enable GT-DCA enhanced appearance modeling during rendering.")
    parser.add_argument("--gt_dca_feature_dim", type=int, default=256, help="GT-DCA feature dimension.")
    parser.add_argument("--gt_dca_num_sample_points", type=int, default=8, help="Number of sampling points for GT-DCA deformable sampling.")
    parser.add_argument("--gt_dca_hidden_dim", type=int, default=128, help="Hidden dimension for GT-DCA MLPs.")
    parser.add_argument("--gt_dca_confidence_threshold", type=float, default=0.5, help="Confidence threshold for GT-DCA track points.")
    parser.add_argument("--gt_dca_min_track_points", type=int, default=4, help="Minimum number of track points required for GT-DCA.")
    parser.add_argument("--gt_dca_enable_caching", action="store_true", help="Enable GT-DCA feature caching for performance.")
    parser.add_argument("--gt_dca_dropout_rate", type=float, default=0.1, help="Dropout rate for GT-DCA modules.")
    parser.add_argument("--gt_dca_attention_heads", type=int, default=8, help="Number of attention heads for GT-DCA cross-attention.")
    # 混合精度选项 - 注意：混合精度现在通过 --mixed_precision 全局控制
    # amp_dtype 参数已移至 OptimizationParams 中统一管理
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Task: 添加GT-DCA启用/禁用的配置选项 - 渲染脚本GT-DCA配置
    # GT-DCA: Setup GT-DCA configuration for rendering
    if hasattr(args, 'use_gt_dca') and args.use_gt_dca:
        from gt_dca.core.data_structures import GTDCAConfig
        
        # Create GT-DCA configuration from command line arguments
        gt_dca_config = GTDCAConfig(
            feature_dim=getattr(args, 'gt_dca_feature_dim', 256),
            hidden_dim=getattr(args, 'gt_dca_hidden_dim', 128),
            num_sample_points=getattr(args, 'gt_dca_num_sample_points', 8),
            confidence_threshold=getattr(args, 'gt_dca_confidence_threshold', 0.5),
            min_track_points=getattr(args, 'gt_dca_min_track_points', 4),
            enable_caching=getattr(args, 'gt_dca_enable_caching', False),
            dropout_rate=getattr(args, 'gt_dca_dropout_rate', 0.1),
            attention_heads=getattr(args, 'gt_dca_attention_heads', 8),
            use_mixed_precision=getattr(args, 'mixed_precision', False),
            amp_dtype=getattr(args, 'amp_dtype', 'fp16')
        )
        
        # Attach GT-DCA configuration to args
        args.gt_dca_config = gt_dca_config
        
        print(f"✅ GT-DCA渲染配置已设置: {gt_dca_config}")
    else:
        args.use_gt_dca = False
        args.gt_dca_config = None

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)
