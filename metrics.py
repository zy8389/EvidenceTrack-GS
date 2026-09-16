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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from utils.depth_utils import estimate_depth

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(gt_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readDepthMaps(renders_dir, gt_dir=None):
    """读取深度图，支持.npy和.png格式"""
    rendered_depths = []
    gt_depths = []
    depth_names = []
    
    # 查找渲染的深度图
    for fname in os.listdir(renders_dir):
        if fname.endswith('_depth.npy'):
            depth_path = renders_dir / fname
            depth = np.load(depth_path)
            rendered_depths.append(torch.from_numpy(depth).cuda())
            depth_names.append(fname.replace('_depth.npy', ''))
    
    # 如果有GT深度图目录，加载GT深度
    if gt_dir and os.path.exists(gt_dir):
        for depth_name in depth_names:
            gt_depth_path = gt_dir / f"{depth_name}_depth.npy"
            if os.path.exists(gt_depth_path):
                gt_depth = np.load(gt_depth_path)
                gt_depths.append(torch.from_numpy(gt_depth).cuda())
            else:
                gt_depths.append(None)
    
    return rendered_depths, gt_depths, depth_names

def compute_depth_metrics(pred_depth, gt_depth=None, gt_image=None):
    """计算深度质量指标"""
    metrics = {}
    
    if gt_depth is not None:
        # 与GT深度图的比较
        valid_mask = (gt_depth > 0) & (pred_depth > 0)
        if valid_mask.sum() > 0:
            pred_valid = pred_depth[valid_mask]
            gt_valid = gt_depth[valid_mask]
            
            metrics['depth_mae'] = torch.mean(torch.abs(pred_valid - gt_valid)).item()
            metrics['depth_rmse'] = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2)).item()
            metrics['depth_abs_rel'] = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()
            
            # 计算深度一致性指标
            delta = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
            metrics['depth_delta1'] = (delta < 1.25).float().mean().item()
            metrics['depth_delta2'] = (delta < 1.25 ** 2).float().mean().item()
            metrics['depth_delta3'] = (delta < 1.25 ** 3).float().mean().item()
    
    elif gt_image is not None:
        # 使用MiDaS估计GT深度进行比较
        with torch.no_grad():
            midas_depth = estimate_depth(gt_image.squeeze(0), mode='test')
            
            # 归一化到相同尺度
            pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
            midas_norm = (midas_depth - midas_depth.min()) / (midas_depth.max() - midas_depth.min())
            
            metrics['depth_midas_mae'] = torch.mean(torch.abs(pred_norm - midas_norm)).item()
            metrics['depth_midas_rmse'] = torch.sqrt(torch.mean((pred_norm - midas_norm) ** 2)).item()
    
    # 深度图的结构化指标
    grad_x = torch.abs(pred_depth[:, 1:] - pred_depth[:, :-1])
    grad_y = torch.abs(pred_depth[1:, :] - pred_depth[:-1, :])
    metrics['depth_smoothness'] = (grad_x.mean() + grad_y.mean()).item() / 2
    
    return metrics

def compute_geometric_metrics(points1, points2=None):
    """计算几何精度指标"""
    metrics = {}
    
    if points2 is not None:
        # Chamfer距离
        dist_1_to_2 = torch.cdist(points1, points2).min(dim=1)[0]
        dist_2_to_1 = torch.cdist(points2, points1).min(dim=1)[0]
        
        metrics['chamfer_distance'] = (dist_1_to_2.mean() + dist_2_to_1.mean()).item() / 2
        metrics['hausdorff_distance'] = max(dist_1_to_2.max().item(), dist_2_to_1.max().item())
    
    # 点云密度分析
    if len(points1) > 1:
        pairwise_dist = torch.cdist(points1, points1)
        pairwise_dist[pairwise_dist == 0] = float('inf')  # 排除自身
        nearest_dist = pairwise_dist.min(dim=1)[0]
        metrics['point_density_mean'] = nearest_dist.mean().item()
        metrics['point_density_std'] = nearest_dist.std().item()
    
    return metrics

def compute_multiview_consistency(renders, depths=None, camera_poses=None):
    """计算多视角一致性指标"""
    metrics = {}
    
    if len(renders) < 2:
        return metrics
    
    # 光度一致性 (Photometric Consistency)
    photometric_errors = []
    for i in range(len(renders)):
        for j in range(i+1, min(i+5, len(renders))):  # 只比较邻近视角
            render_i = renders[i].squeeze(0)
            render_j = renders[j].squeeze(0)
            
            # 计算图像间的SSIM作为光度一致性指标
            photo_error = 1 - ssim(renders[i], renders[j]).item()
            photometric_errors.append(photo_error)
    
    if photometric_errors:
        metrics['photometric_consistency'] = 1 - np.mean(photometric_errors)
    
    # 深度一致性 (如果有深度图)
    if depths is not None and len(depths) >= 2:
        depth_consistency_errors = []
        for i in range(len(depths)):
            for j in range(i+1, min(i+3, len(depths))):
                # 简单的深度差异计算
                depth_diff = torch.abs(depths[i] - depths[j]).mean()
                depth_consistency_errors.append(depth_diff.item())
        
        if depth_consistency_errors:
            # 一致性越高，误差越小
            metrics['depth_consistency'] = 1 / (1 + np.mean(depth_consistency_errors))
    
    return metrics

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 基础图像质量指标
                ssims = []
                psnrs = []
                lpipss = []
                
                # 深度相关指标
                depth_maes = []
                depth_rmses = []
                depth_midas_maes = []
                depth_smoothness_scores = []
                depth_delta1_scores = []

                # 尝试加载深度图
                rendered_depths, gt_depths, depth_names = readDepthMaps(renders_dir, gt_dir)
                
                print(f"  Found {len(rendered_depths)} depth maps")

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # 原有的图像质量指标
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    
                    # 深度指标（如果有对应的深度图）
                    if idx < len(rendered_depths):
                        depth_metrics = compute_depth_metrics(
                            rendered_depths[idx], 
                            gt_depths[idx] if idx < len(gt_depths) else None,
                            gts[idx]
                        )
                        
                        if 'depth_mae' in depth_metrics:
                            depth_maes.append(depth_metrics['depth_mae'])
                            depth_rmses.append(depth_metrics['depth_rmse'])
                            depth_delta1_scores.append(depth_metrics['depth_delta1'])
                        elif 'depth_midas_mae' in depth_metrics:
                            depth_midas_maes.append(depth_metrics['depth_midas_mae'])
                        
                        depth_smoothness_scores.append(depth_metrics['depth_smoothness'])

                # 打印结果
                print("RGB Quality Metrics:")
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                
                if depth_maes:
                    print("Depth Quality Metrics (vs GT):")
                    print("  Depth MAE  : {:>12.7f}".format(np.mean(depth_maes), ".5"))
                    print("  Depth RMSE : {:>12.7f}".format(np.mean(depth_rmses), ".5"))
                    print("  Depth δ<1.25: {:>12.7f}".format(np.mean(depth_delta1_scores), ".5"))
                elif depth_midas_maes:
                    print("Depth Quality Metrics (vs MiDaS):")
                    print("  Depth MAE (MiDaS): {:>12.7f}".format(np.mean(depth_midas_maes), ".5"))
                
                if depth_smoothness_scores:
                    print("  Depth Smoothness: {:>12.7f}".format(np.mean(depth_smoothness_scores), ".5"))
                
                print("")

                # 保存指标
                metrics_dict = {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()
                }
                
                if depth_maes:
                    metrics_dict.update({
                        "Depth_MAE": np.mean(depth_maes),
                        "Depth_RMSE": np.mean(depth_rmses),
                        "Depth_Delta1": np.mean(depth_delta1_scores)
                    })
                elif depth_midas_maes:
                    metrics_dict["Depth_MAE_MiDaS"] = np.mean(depth_midas_maes)
                
                if depth_smoothness_scores:
                    metrics_dict["Depth_Smoothness"] = np.mean(depth_smoothness_scores)
                
                full_dict[scene_dir][method].update(metrics_dict)
                
                # Per-view指标
                per_view_metrics = {
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                }
                
                if depth_smoothness_scores:
                    per_view_metrics["Depth_Smoothness"] = {
                        name: smooth for smooth, name in zip(depth_smoothness_scores, image_names[:len(depth_smoothness_scores)])
                    }
                
                per_view_dict[scene_dir][method].update(per_view_metrics)
                
                # 计算多视角一致性指标
                if len(renders) > 1:
                    multiview_metrics = compute_multiview_consistency(
                        renders, 
                        rendered_depths if rendered_depths else None
                    )
                    
                    if multiview_metrics:
                        print("Multi-view Consistency Metrics:")
                        for key, value in multiview_metrics.items():
                            print(f"  {key.replace('_', ' ').title()}: {value:>12.7f}")
                        print("")
                        
                        # 添加到结果字典
                        full_dict[scene_dir][method].update(multiview_metrics)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', '-s', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()
    # if not os.path.exists(os.path.join(args.model_paths[0], 'test', 'ours_'+str(args.iteration), 'renders')):
    # os.system('python render.py -s /mnt/vita-nas/zehao/nerf_llff_data/horns/ --iteration ' + str(args.iteration) + ' -m ' +args.model_paths[0])
    evaluate(args.model_paths)
