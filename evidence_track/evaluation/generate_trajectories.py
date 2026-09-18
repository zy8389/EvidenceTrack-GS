# -*- coding: utf-8 -*-
"""
本脚本用于读取 Colmap 重建生成的 `points3D.bin` 文件，
并将其中的 3D 点及其对应的观测轨迹（tracks）转换为 HDF5 (.h5) 格式。

Colmap 的 `points3D.bin` 文件包含了每个三维点的 XYZ 坐标、颜色、误差
以及一个"轨迹"列表，该列表指明了此三维点在哪些图像中被看到，
以及对应的二维关键点索引。

输出的 `tracks.h5` 文件将包含以下数据集：
- `points3D`: (N, 3) 大小的数组，存储每个 3D 点的 XYZ 坐标。
- `point3D_ids`: (N,) 大小的数组，存储每个 3D 点在 Colmap 中的原始 ID。
- `track_lengths`: (N,) 大小的数组，存储每个 3D 点的轨迹长度（即被多少个相机观测到）。
- `image_ids`: (T,) 大小的数组，其中 T 是所有轨迹长度的总和。它将所有点的观测图像 ID 连接成一个长数组。
- `point2D_idxs`: (T,) 大小的数组，存储与 `image_ids` 对应的二维关键点索引。

如何使用:
1. 确保已安装 numpy 和 h5py:
   pip install numpy h5py
2. 在终端中运行此脚本，并指定 Colmap 的 sparse/0 文件夹路径:
   python your_script_name.py --colmap_path /path/to/your/project/sparse/0
3. 脚本将在指定的 Colmap 路径下生成一个 `tracks.h5` 文件。
"""

import os
import argparse
import struct
import collections
import numpy as np
import h5py

# 定义一个具名元组来存储解析后的 3D 点信息，方便访问
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_points3d_binary(path_to_model_file):
    """
    解析 Colmap 的 points3D.bin 文件。
    
    Colmap 二进制文件格式参考: https://colmap.github.io/format.html
    
    :param path_to_model_file: `points3D.bin` 文件的路径。
    :return: 一个字典，键是 point3D_id，值是 Point3D 对象。
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        # 读取文件头的 3D 点数量
        num_points = struct.unpack('<Q', fid.read(8))[0]
        
        # 循环读取每个点的信息
        for _ in range(num_points):
            # 读取点ID, XYZ, RGB, Error
            binary_point_line = fid.read(43) # 8 + 3*8 + 3*1 + 8 = 43 bytes
            
            # 修复: 移除了格式字符串中无效的 '_' 字符
            point3D_id, x, y, z, r, g, b, error = struct.unpack('<QdddBBBd', binary_point_line)
            
            # 读取轨迹长度
            track_len = struct.unpack('<Q', fid.read(8))[0]
            
            # 读取轨迹数据 (image_id, point2D_idx)
            track_data = fid.read(8 * track_len)
            track = struct.unpack('<' + 'II' * track_len, track_data)
            
            # 将轨迹数据解包为 image_ids 和 point2D_idxs
            image_ids = np.array(track[0::2])
            point2D_idxs = np.array(track[1::2])
            
            # 存入字典
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=np.array((x, y, z)),
                rgb=np.array((r, g, b)),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs)
            
    return points3D

def main():
    """
    主函数，负责解析参数、调用读取函数并生成 HDF5 文件。
    """
    parser = argparse.ArgumentParser(description="从 Colmap 的 sparse/0 目录生成 tracks.h5 文件。")
    parser.add_argument("--colmap_path", required=True,
                        help="Colmap sparse/0 文件夹的路径。")
    parser.add_argument("--output_file", default="tracks.h5",
                        help="输出的 HDF5 文件名 (默认: tracks.h5)。")
    
    args = parser.parse_args()

    colmap_path = args.colmap_path
    points_bin_path = os.path.join(colmap_path, "points3D.bin")
    output_h5_path = os.path.join(colmap_path, args.output_file)

    if not os.path.exists(points_bin_path):
        print(f"错误: 在路径 '{colmap_path}' 中未找到 'points3D.bin' 文件。")
        return

    print(f"正在读取 Colmap 模型: {points_bin_path}")
    try:
        points3D_data = read_points3d_binary(points_bin_path)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
        
    print(f"成功读取 {len(points3D_data)} 个 3D 点。")

    # 将数据整理成扁平化的 numpy 数组，以便存入 HDF5
    # 按照 point ID 排序以保证确定性
    sorted_ids = sorted(points3D_data.keys())
    
    all_points_xyz = []
    all_point_ids = []
    all_track_lengths = []
    concatenated_image_ids = []
    concatenated_point2D_idxs = []

    for point_id in sorted_ids:
        point = points3D_data[point_id]
        all_points_xyz.append(point.xyz)
        all_point_ids.append(point.id)
        all_track_lengths.append(len(point.image_ids))
        concatenated_image_ids.extend(point.image_ids)
        concatenated_point2D_idxs.extend(point.point2D_idxs)

    # 转换为 Numpy 数组
    points_xyz_np = np.array(all_points_xyz, dtype=np.float64)
    point_ids_np = np.array(all_point_ids, dtype=np.uint64)
    track_lengths_np = np.array(all_track_lengths, dtype=np.int32)
    image_ids_np = np.array(concatenated_image_ids, dtype=np.uint32)
    point2D_idxs_np = np.array(concatenated_point2D_idxs, dtype=np.uint32)

    print(f"正在将轨迹数据写入 HDF5 文件: {output_h5_path}")
    try:
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('points3D', data=points_xyz_np)
            f.create_dataset('point3D_ids', data=point_ids_np)
            f.create_dataset('track_lengths', data=track_lengths_np)
            f.create_dataset('image_ids', data=image_ids_np)
            f.create_dataset('point2D_idxs', data=point2D_idxs_np)
            
            # 写入一些元数据
            f.attrs['num_points'] = len(points3D_data)
            f.attrs['total_observations'] = len(image_ids_np)

    except Exception as e:
        print(f"写入 HDF5 文件时发生错误: {e}")
        return

    print("处理完成！")
    print(f"文件已保存到: {output_h5_path}")
    print("\n如何从 tracks.h5 文件中读取第 i 个点的轨迹示例:")
    print("""
import h5py
import numpy as np

with h5py.File('""" + output_h5_path + """', 'r') as f:
    track_lengths = f['track_lengths'][:]
    image_ids = f['image_ids'][:]
    
    # 计算每个轨迹的起始索引
    track_starts = np.cumsum(np.concatenate(([0], track_lengths[:-1]))).astype(int)
    
    # 示例：获取索引为 100 的点的轨迹
    i = 100
    if i < len(track_lengths):
        start_idx = track_starts[i]
        end_idx = start_idx + track_lengths[i]
        point_track_image_ids = image_ids[start_idx:end_idx]
        print(f"点 {i} 的轨迹长度为: {track_lengths[i]}")
        print(f"点 {i} 的观测图像 ID: {point_track_image_ids}")
""")

if __name__ == "__main__":
    main()