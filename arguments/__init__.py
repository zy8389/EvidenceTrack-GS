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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 0
        # --- EvidenceTrack-GS model and data parameters ---
        self.track_path = "tracks.h5"  # 预计算的特征轨迹文件路径
        # Phase 2.1 protocols.  Legacy remains the default for benchmark
        # reproduction; strict mode fails closed unless a v4 source-RGB-membership H5 is
        # supplied and its source-image set exactly matches the training split.
        self.strict_tracks = False
        self.strict_source_only_geometry = False
        self.use_gtdca_attention = False  # 启用GT-DCA注意力模块
        self.enable_geometric_constraints = False  # 启用几何约束系统
        self.constraint_config_path = ""  # 约束配置文件路径
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 300
        self.prune_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0005
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 10  # 减少伪相机采样频率
        self.dist_thres = 10.
        self.depth_weight = 0.03  # 降低深度损失权重加速计算
        self.depth_pseudo_weight = 0.5

        # Phase 2.1 repaired source-only geometry.
        self.strict_track_weight = 0.1
        self.strict_track_start = 500
        self.strict_track_warmup = 1000
        self.strict_track_min_length = 2
        self.strict_track_min_quality = 0.05
        self.strict_track_max_tracks = 0
        self.strict_track_huber_delta = 2.0
        self.strict_track_association_chunk_size = 1024
        self.strict_track_max_association_distance_ratio = 0.05
        self.strict_track_log_interval = 100
        self.strict_track_collision_warning_rate = 0.25
        self.strict_track_gradient_test_count = 8
        self.strict_track_min_gradient_coverage = 0.95
        self.strict_track_gradient_probe_offset_ratio = 0.0001
        self.geometry_smoke_only = False

        # Controlled A0/A1/B continuation.  Real-view and pseudo-view sampling
        # use separate deterministic schedules.
        self.experiment_seed = 1
        self.controlled_ab_role = ""  # empty, A0, A1, SelfRender, or B
        self.controlled_ab_checkpoint_iteration = 10000

        # Frozen, manifest-driven Difix pseudo RGB.  No diffusion module is
        # imported into the EvidenceTrack-GS training environment.
        self.enable_diffusion_pseudo_rgb = False
        self.pseudo_rgb_manifest = ""
        self.pseudo_rgb_weight = 0.10
        self.pseudo_rgb_lambda_dssim = 0.20
        self.pseudo_rgb_start = 10000
        self.pseudo_rgb_end = 12000
        self.pseudo_rgb_interval = 50
        self.pseudo_rgb_strict_cache = False
        self.disable_legacy_pseudo_depth = False
        # --- EvidenceTrack-GS optimization and loss parameters ---
        self.use_hybrid_loss = False  # 启用混合几何损失模型
        self.disable_depth_loss = False  # 完全禁用原有的深度损失 (用于消融实验)
        self.lambda_reproj = 0.1  # [如果不用动态加权] 全局重投影损失的静态权重
        
        # 几何约束权重参数
        self.geometric_constraint_weight = 0.1  # 几何约束损失权重
        self.multiscale_constraint_weight = 0.05  # 多尺度约束权重
        self.consistency_constraint_weight = 0.02  # 一致性约束权重
        
        # 自适应权重参数
        self.enable_adaptive_weighting = True  # 启用自适应权重
        self.texture_weight_min = 0.3  # 纹理权重最小值
        self.texture_weight_max = 2.0  # 纹理权重最大值
        self.confidence_decay_factor = 2.0  # 置信度衰减因子
        
        # 质量评估参数
        self.min_trajectory_quality = 0.4  # 最小轨迹质量阈值
        self.max_outlier_ratio = 0.3  # 最大异常值比例
        self.outlier_threshold_pixels = 2.0  # 异常值阈值（像素）
        
        # 多尺度参数
        self.multiscale_scales = [1.0, 0.5, 0.25]  # 多尺度比例
        self.multiscale_weights = [0.5, 0.3, 0.2]  # 多尺度权重
        
        # 验证参数
        self.constraint_validation_interval = 100  # 约束验证间隔
        self.constraint_satisfaction_threshold = 0.85  # 约束满足度阈值
        
        # 鲁棒损失参数
        self.robust_loss_type = "huber"  # 鲁棒损失类型 ("huber", "l1", "l2")
        self.huber_delta = 1.0  # Huber损失阈值参数
        
        # 几何正则化参数
        self.geometry_reg_enabled = False  # 启用几何正则化
        self.geometry_reg_weight = 0.01  # 几何正则化权重
        self.geometry_reg_k_neighbors = 16  # PCA分析的邻居数量
        self.geometry_reg_enable_threshold = 5000  # 开始正则化的迭代阈值
        self.geometry_reg_min_eigenvalue_ratio = 0.1  # 最小特征值比率
        
        # 混合精度训练参数
        self.mixed_precision = False  # 启用整个训练流程的混合精度（AMP）
        self.amp_dtype = "fp16"  # AMP精度类型 (fp16 或 bf16)
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
