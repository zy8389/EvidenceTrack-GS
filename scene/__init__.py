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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
from scene.cameras import PseudoCamera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        # ==================================================================
        # vvvvvvvv 这是第一个关键修复：使用 --data_type 参数 vvvvvvvv
        # ==================================================================
        # 不再通过检查文件路径来猜测，而是直接使用我们定义的类型
        if args.data_type == "colmap" or args.data_type == "360":
            # 360度数据集也是COLMAP格式的一种
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                args.n_views,
                getattr(args, "llff_holdout", 8),
                getattr(args, "strict_source_only_geometry", False),
                getattr(args, "track_path", None),
            )
        elif args.data_type == "blender":
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, f"Could not recognize scene type '{args.data_type}'!"
        # ==================================================================
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ==================================================================


        if not self.loaded_iter:
            # 如果 scene_info 为 None, 可能是加载失败
            if not scene_info:
                 assert False, f"Failed to load scene info for data type '{args.data_type}' at path '{args.source_path}'"
            
            # 确保点云路径存在
            if scene_info.ply_path and os.path.exists(scene_info.ply_path):
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            elif getattr(args, "strict_source_only_geometry", False):
                from geometric_constraints.strict_track_store import (
                    TRACK_MEMBERSHIP_PROVENANCE,
                )

                strict_input_path = os.path.join(self.model_path, "input.ply")
                storePly(
                    strict_input_path,
                    np.asarray(scene_info.point_cloud.points, dtype=np.float32),
                    np.clip(
                        np.asarray(scene_info.point_cloud.colors) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8),
                )
                with open(
                    os.path.join(self.model_path, "point_cloud_provenance.json"),
                    "w",
                    encoding="utf-8",
                ) as provenance_file:
                    json.dump(
                        {
                            "protocol": "strict_source_only",
                            "source": os.path.abspath(args.track_path),
                            "provenance": "source_only_track_anchors_and_source_rgb",
                            "track_membership_provenance": (
                                TRACK_MEMBERSHIP_PROVENANCE
                            ),
                            "full_colmap_point_cloud_used": False,
                        },
                        provenance_file,
                        indent=2,
                    )
            else:
                # 如果没有点云，创建一个空的 input.ply 文件，避免崩溃
                print(f"[Warning] scene_info.ply_path not found at '{scene_info.ply_path}'. Creating empty input.ply.")
                with open(os.path.join(self.model_path, "input.ply"), 'w') as f:
                    pass # 创建空文件

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            # ==================================================================
            # vvvvvvvv 这是第二个关键修复：修复伪相机生成逻辑 vvvvvvvv
            # ==================================================================
            if len(self.train_cameras[resolution_scale]) > 0: # 确保有训练相机
                # 使用我们明确的 data_type 来判断
                if args.data_type == '360':
                    pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
                else: # 默认为 llff 风格的生成方式
                    pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
                
                view = self.train_cameras[resolution_scale][0]
                for pose in pseudo_poses:
                    pseudo_cams.append(PseudoCamera(
                        R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                        width=view.image_width, height=view.image_height
                    ))
            # ==================================================================
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # ==================================================================
            self.pseudo_cameras[resolution_scale] = pseudo_cams


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
