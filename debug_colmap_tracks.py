#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug COLMAP tracks.h5 文件结构
帮助理解真实的数据格式
"""

import h5py
import numpy as np
import sys

def debug_colmap_h5(h5_path: str):
    """详细分析COLMAP H5文件结构"""
    print(f"分析文件: {h5_path}")
    print("=" * 80)
    
    with h5py.File(h5_path, 'r') as f:
        print("根级别keys:")
        for key in f.keys():
            data = f[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
            
            # 显示前几个值
            if data.size < 20:
                print(f"    完整数据: {data[:]}")
            else:
                print(f"    前10个值: {data[:10]}")
                if len(data.shape) == 2:
                    print(f"    第一行: {data[0]}")
            print()
        
        # 特别分析image_ids和point2D_idxs的关系
        if 'image_ids' in f and 'point2D_idxs' in f and 'track_lengths' in f:
            image_ids = f['image_ids'][:]
            point2D_idxs = f['point2D_idxs'][:]
            track_lengths = f['track_lengths'][:]
            
            print(f"轨迹分析:")
            print(f"  总观测数: {len(image_ids)}")
            print(f"  总轨迹数: {len(track_lengths)}")
            print(f"  图像ID范围: {image_ids.min()} - {image_ids.max()}")
            print(f"  Point2D索引范围: {point2D_idxs.min()} - {point2D_idxs.max()}")
            print()
            
            # 分析前几条轨迹
            print("前5条轨迹详情:")
            start_idx = 0
            for i, length in enumerate(track_lengths[:5]):
                end_idx = start_idx + length
                track_img_ids = image_ids[start_idx:end_idx]
                track_pt2d_ids = point2D_idxs[start_idx:end_idx]
                
                print(f"  轨迹 {i}: 长度={length}")
                print(f"    图像IDs: {track_img_ids}")
                print(f"    Point2D索引: {track_pt2d_ids}")
                
                start_idx = end_idx
            print()
        
        # 分析points3D数据
        if 'points3D' in f:
            points3d = f['points3D'][:]
            print(f"3D点数据:")
            print(f"  形状: {points3d.shape}")
            print(f"  前3个点:")
            for i in range(min(3, points3d.shape[0])):
                print(f"    点{i}: {points3d[i]}")
            print()


def suggest_coordinate_extraction(h5_path: str):
    """建议如何提取真实的2D坐标"""
    print("建议的坐标提取方法:")
    print("=" * 40)
    
    print("COLMAP tracks.h5文件通常不包含实际的2D像素坐标。")
    print("它只包含以下信息:")
    print("- image_ids: 图像ID")  
    print("- point2D_idxs: 每个图像中keypoint的索引")
    print("- point3D_ids: 对应的3D点ID")
    print("- track_lengths: 每条轨迹的长度")
    print()
    
    print("要获得真实的2D像素坐标，你需要:")
    print("1. COLMAP的images.bin文件 (包含每个图像的keypoints)")
    print("2. 或者原始的特征检测结果")
    print("3. 或者使用COLMAP的Python API重新提取")
    print()
    
    print("当前解决方案:")
    print("由于没有真实的2D坐标，我们将:")
    print("- 使用智能的随机坐标生成")
    print("- 基于track_id确保坐标一致性")
    print("- 提供合理的图像坐标范围")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python debug_colmap_tracks.py /path/to/tracks.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    debug_colmap_h5(h5_path)
    suggest_coordinate_extraction(h5_path)