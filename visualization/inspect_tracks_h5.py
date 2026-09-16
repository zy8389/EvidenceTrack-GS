#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查tracks.h5文件结构的脚本
用于了解数据格式，为后续的2D和3D可视化做准备
"""

import h5py
import numpy as np
import argparse
from pathlib import Path

def inspect_h5_structure(h5_path: str):
    """详细检查H5文件的结构和内容"""
    print(f"检查H5文件: {h5_path}")
    print("=" * 60)
    
    with h5py.File(h5_path, 'r') as f:
        print(f"文件根级别的keys: {list(f.keys())}")
        print()
        
        def print_structure(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name}")
                print(f"{indent}   Keys: {list(obj.keys())}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}")
                print(f"{indent}   Dtype: {obj.dtype}")
                print(f"{indent}   Size: {obj.size}")
                
                # 如果数据集不太大，显示一些样本数据
                if obj.size < 1000:
                    print(f"{indent}   Data preview: {obj[...]}")
                elif len(obj.shape) == 2 and obj.shape[0] < 10:
                    print(f"{indent}   First few rows:")
                    for i in range(min(5, obj.shape[0])):
                        print(f"{indent}     Row {i}: {obj[i]}")
                elif len(obj.shape) == 1:
                    print(f"{indent}   First 10 values: {obj[:10]}")
                else:
                    print(f"{indent}   First element: {obj.flat[0] if obj.size > 0 else 'Empty'}")
            print()
        
        # 递归遍历所有项目
        f.visititems(print_structure)
        
        # 特别检查一些常见的轨迹数据字段
        common_keys = ['tracks', 'points2D', 'points3D', 'images', 'cameras', 'keypoints', 'matches']
        print("🔍 检查常见的轨迹数据字段:")
        for key in common_keys:
            if key in f:
                obj = f[key]
                print(f"  ✅ {key}: Shape={obj.shape if hasattr(obj, 'shape') else 'Group'}, Type={type(obj)}")
            else:
                print(f"  ❌ {key}: Not found")
        print()

def analyze_track_data(h5_path: str):
    """分析轨迹数据的特性"""
    print("🎯 轨迹数据分析:")
    print("-" * 40)
    
    with h5py.File(h5_path, 'r') as f:
        # 尝试找到轨迹数据
        possible_track_keys = ['tracks', 'trajectories', 'track_data']
        track_data = None
        track_key = None
        
        for key in possible_track_keys:
            if key in f:
                track_data = f[key]
                track_key = key
                break
        
        if track_data is not None:
            print(f"找到轨迹数据: {track_key}")
            print(f"数据形状: {track_data.shape}")
            print(f"数据类型: {track_data.dtype}")
            
            # 分析轨迹统计信息
            if len(track_data.shape) >= 2:
                print(f"轨迹数量: {track_data.shape[0]}")
                if track_data.shape[1] >= 2:
                    print("前5个轨迹的样本数据:")
                    for i in range(min(5, track_data.shape[0])):
                        print(f"  轨迹 {i}: {track_data[i][:min(10, track_data.shape[1])]}...")
        else:
            print("未找到明确的轨迹数据字段")
        
        # 检查2D点数据
        points2d_keys = ['points2D', 'keypoints', 'features', 'observations']
        for key in points2d_keys:
            if key in f:
                points2d = f[key]
                print(f"\n找到2D点数据: {key}")
                print(f"形状: {points2d.shape}")
                print(f"类型: {points2d.dtype}")
                if points2d.size > 0:
                    print(f"数据范围: X=[{np.min(points2d[..., 0]):.1f}, {np.max(points2d[..., 0]):.1f}], "
                          f"Y=[{np.min(points2d[..., 1]):.1f}, {np.max(points2d[..., 1]):.1f}]")
                break
        
        # 检查3D点数据
        points3d_keys = ['points3D', 'points_3d', 'structure', 'landmarks']
        for key in points3d_keys:
            if key in f:
                points3d = f[key]
                print(f"\n找到3D点数据: {key}")
                print(f"形状: {points3d.shape}")
                print(f"类型: {points3d.dtype}")
                if points3d.size > 0 and len(points3d.shape) >= 2 and points3d.shape[-1] >= 3:
                    print(f"3D点数量: {points3d.shape[0] if len(points3d.shape) == 2 else 'Multiple tracks'}")
                    print(f"坐标范围: X=[{np.min(points3d[..., 0]):.2f}, {np.max(points3d[..., 0]):.2f}], "
                          f"Y=[{np.min(points3d[..., 1]):.2f}, {np.max(points3d[..., 1]):.2f}], "
                          f"Z=[{np.min(points3d[..., 2]):.2f}, {np.max(points3d[..., 2]):.2f}]")
                break

def main():
    parser = argparse.ArgumentParser(description="检查tracks.h5文件结构")
    parser.add_argument("h5_path", type=str, help="tracks.h5文件路径")
    
    args = parser.parse_args()
    
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        print(f"错误: 文件 {h5_path} 不存在")
        return
    
    # 检查文件结构
    inspect_h5_structure(str(h5_path))
    
    # 分析轨迹数据
    analyze_track_data(str(h5_path))
    
    print("\n" + "=" * 60)
    print("检查完成! 请根据上述信息准备2D和3D可视化脚本。")

if __name__ == "__main__":
    main()