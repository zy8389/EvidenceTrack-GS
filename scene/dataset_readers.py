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
import glob
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.general_utils import chamfer_dist
import numpy as np
import json
import cv2
import math
import torch
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

        break

    def normalize(x):
        return x / np.linalg.norm(x)

    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m


    c2w = np.concatenate([R, T], dim=1)
    print(c2w.shape)
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    Num_views = 120
    rots = 2

    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, Num_views + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))


    sys.stdout.write('\n')
    return cam_infos

def readColmap360Cameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    一个专门为标准COLMAP输出（特别是360度场景）设计的、稳健的相机加载器。
    它不依赖任何 poses_bounds.npy 文件。
    """
    cam_infos = []
    # 遍历从COLMAP读取的所有相机外参
    for idx, key in enumerate(tqdm(cam_extrinsics.keys(), desc="Reading 360 cameras")):
        extr = cam_extrinsics[key]
        
        # 检查这个相机对应的图像文件是否真实存在
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            # 如果文件不存在，则跳过这个相机，并打印警告
            # 这使得加载器非常稳健，不会因为模型与文件列表不匹配而崩溃
            # print(f"\n[Warning] Could not find image: {image_path}, skipping camera.")
            continue

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # 从COLMAP内参计算FOV (视场角)
        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, f"Colmap camera model not handled: {intr.model}"

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # 这是关键修复点：不再加载 poses_bounds.npy
        # 我们为360度无边界场景设置固定的远近裁剪平面
        near_plane = 0.01
        far_plane = 100.0
        bounds = np.array([near_plane, far_plane])

        # 创建并存储相机信息
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              mask=None, bounds=bounds) # mask 设为 None
        cam_infos.append(cam_info)
        
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=None, bounds=bounds)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def farthest_point_sampling(points, k):
    """
    Sample k points from input pointcloud data points using Farthest Point Sampling.

    Parameters:
    points: numpy.ndarray
        The input pointcloud data, a numpy array of shape (N, D) where N is the
        number of points and D is the dimensionality of each point.
    k: int
        The number of points to sample.

    Returns:
    sampled_points: numpy.ndarray
        The sampled pointcloud data, a numpy array of shape (k, D).
    """
    N, D = points.shape
    farthest_pts = np.zeros((k, D))
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(k):
        farthest_pts[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return farthest_pts


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path,
    images,
    eval,
    n_views=0,
    llffhold=8,
    strict_source_only_geometry=False,
    track_path=None,
):
    # --- 1. 动态寻找 COLMAP sparse 模型路径 ---
    # 首先确定基础的 sparse 目录是否存在
    base_sparse_dir = os.path.join(path, "sparse")
    if not os.path.isdir(base_sparse_dir):
        print(f"❌ [ERROR]: 'sparse' directory not found in '{path}'")
        # 如果连sparse目录都没有，就无法继续
        return None 

    # 检查是否存在 '0' 子目录，如果存在，则使用它；否则，直接使用 'sparse' 目录
    colmap_model_path = ""
    if os.path.isdir(os.path.join(base_sparse_dir, "0")):
        print("✅ [INFO]: Found 'sparse/0' subdirectory, using it as the model path.")
        colmap_model_path = os.path.join(base_sparse_dir, "0")
    else:
        print("✅ [INFO]: No 'sparse/0' subdirectory found, using 'sparse' as the model path.")
        colmap_model_path = base_sparse_dir

    # --- 2. 加载相机参数 ---
    # 使用动态确定的 colmap_model_path
    try:
        cameras_extrinsic_file = os.path.join(colmap_model_path, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_model_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except FileNotFoundError:
        print(" Binary files not found, trying text files...")
        cameras_extrinsic_file = os.path.join(colmap_model_path, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_model_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # --- 3. 加载点云 ---
    # Legacy protocol loads the full COLMAP point cloud.  Strict protocol must
    # not even materialize it: its Gaussian initialisation is built later from
    # source-only re-triangulated anchors in the strict H5 file.
    ply_path = os.path.join(colmap_model_path, "points3D.ply")
    bin_path = os.path.join(colmap_model_path, "points3D.bin")
    if strict_source_only_geometry:
        print("[Geometry Protocol] STRICT SOURCE-ONLY PROTOCOL")
        print("[Strict Geometry] Skipping full points3D.bin/points3D.ply initialisation")
        ply_path = None
        pcd = None
    else:
        print(
            "[Geometry Protocol] LEGACY PROTOCOL: full COLMAP point-cloud "
            "geometry may include held-out-image contributions"
        )
        if not os.path.exists(ply_path) and os.path.exists(bin_path):
            try:
                print(f"Converting .bin to .ply for '{bin_path}'...")
                xyz, rgb, _ = read_points3D_binary(bin_path)
                storePly(ply_path, xyz, rgb)
            except Exception as e:
                print(f"Could not convert .bin to .ply: {e}")
                ply_path = None

    if not strict_source_only_geometry:
        pcd = None
        if ply_path and os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
        else:
            # 如果没有点云，创建一个空的
            print(f"⚠️ [WARNING]: Point cloud not found. Continuing without it.")
            pcd = BasicPointCloud(points=np.empty((0,3)), colors=np.empty((0,3)), normals=np.empty((0,3)))

    # --- 4. 调用我们新的、正确的相机加载函数 ---
    images_folder = os.path.join(path, images if images is not None else "images")
    cam_infos_unsorted = readColmap360Cameras(cam_extrinsics=cam_extrinsics, 
                                              cam_intrinsics=cam_intrinsics,
                                              images_folder=images_folder)
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    # --- 5. 划分训练/测试集并进行二次采样 (这是我们修改的核心) ---
    if llffhold > 0:
        print(f"✅ [INFO] LLFF holdout factor: {llffhold}. Splitting data into training and testing sets.")
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        print(f"✅ [INFO] Data split: {len(train_cam_infos)} train, {len(test_cam_infos)} test.")
    else:
        print("✅ [INFO] No LLFF holdout. Using all cameras for training.")
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0 and len(train_cam_infos) > n_views:
        print(f"Subsampling training views from {len(train_cam_infos)} to {n_views}")
        indices = np.linspace(0, len(train_cam_infos) - 1, n_views, dtype=int)
        train_cam_infos = [train_cam_infos[i] for i in indices]

    if strict_source_only_geometry:
        if not track_path:
            raise ValueError(
                "--track_path is required by --strict_source_only_geometry"
            )
        from geometric_constraints.strict_track_store import (
            StrictTrackStore,
            normalize_image_name,
        )

        strict_store = StrictTrackStore.load(track_path)
        expected_sources = set(strict_store.source_images)
        actual_sources = {
            normalize_image_name(camera.image_name) for camera in train_cam_infos
        }
        if expected_sources != actual_sources:
            raise RuntimeError(
                "Strict H5 source-image set does not match the training split. "
                f"H5={sorted(expected_sources)}, training={sorted(actual_sources)}"
            )
        audit = strict_store.leakage_audit()
        audit.emit()
        audit.require_pass()
        pcd = BasicPointCloud(
            points=np.asarray(strict_store.xyz, dtype=np.float32),
            colors=np.asarray(strict_store.rgb_source, dtype=np.float32),
            normals=np.zeros_like(strict_store.xyz, dtype=np.float32),
        )
        ply_path = None
        print(
            f"[Strict Geometry] Gaussian initialisation uses {len(strict_store)} "
            "source-only re-triangulated anchors"
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            mask = norm_data[:, :, 3:4]
            if skip == 1:
                depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                                      image_path.split('/')[-4]+'/'+image_name+'_depth.npy')
            else:
                depth_image = None

            arr = cv2.resize(arr, (400, 400))
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            depth_image = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask = None if mask is None else cv2.resize(mask, (400, 400))


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                        image_name=image_name, width=image.size[0], height=image.size[1],
                                        depth_image=depth_image, mask=mask))
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    pseudo_cam_infos = train_cam_infos #train_cam_infos
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 30000
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
