"""
GeoTrack-GS 配置管理工具

提供配置文件的加载、验证和管理功能
"""

import json
import os
from typing import Dict, Any, Optional
from argparse import Namespace


class ConfigValidator:
    """配置参数验证器"""
    
    @staticmethod
    def validate_geometric_constraints_config(args: Namespace) -> bool:
        """
        验证几何约束相关配置参数
        
        Args:
            args: 命令行参数对象
            
        Returns:
            验证是否通过
        """
        errors = []
        
        # 验证权重参数
        if hasattr(args, 'geometric_constraint_weight'):
            if not (0.0 <= args.geometric_constraint_weight <= 1.0):
                errors.append("geometric_constraint_weight must be between 0.0 and 1.0")
        
        if hasattr(args, 'texture_weight_min') and hasattr(args, 'texture_weight_max'):
            if args.texture_weight_min >= args.texture_weight_max:
                errors.append("texture_weight_min must be less than texture_weight_max")
        
        # 验证质量参数
        if hasattr(args, 'min_trajectory_quality'):
            if not (0.0 <= args.min_trajectory_quality <= 1.0):
                errors.append("min_trajectory_quality must be between 0.0 and 1.0")
        
        if hasattr(args, 'max_outlier_ratio'):
            if not (0.0 <= args.max_outlier_ratio <= 1.0):
                errors.append("max_outlier_ratio must be between 0.0 and 1.0")
        
        # 验证多尺度参数
        if hasattr(args, 'multiscale_scales') and hasattr(args, 'multiscale_weights'):
            if len(args.multiscale_scales) != len(args.multiscale_weights):
                errors.append("multiscale_scales and multiscale_weights must have the same length")
            
            if abs(sum(args.multiscale_weights) - 1.0) > 1e-6:
                errors.append("multiscale_weights must sum to 1.0")
        
        # 验证文件路径
        if hasattr(args, 'track_path') and args.track_path:
            if not os.path.exists(args.track_path):
                errors.append(f"Track file not found: {args.track_path}")
        
        if hasattr(args, 'constraint_config_path') and args.constraint_config_path:
            if not os.path.exists(args.constraint_config_path):
                errors.append(f"Constraint config file not found: {args.constraint_config_path}")
        
        # 验证鲁棒损失参数
        if hasattr(args, 'robust_loss_type'):
            if args.robust_loss_type not in ["huber", "l1", "l2"]:
                errors.append("robust_loss_type must be one of: huber, l1, l2")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @staticmethod
    def validate_file_paths(args: Namespace) -> bool:
        """
        验证文件路径的有效性
        
        Args:
            args: 命令行参数对象
            
        Returns:
            验证是否通过
        """
        # 检查源路径
        if hasattr(args, 'source_path') and args.source_path:
            if not os.path.exists(args.source_path):
                print(f"Error: Source path does not exist: {args.source_path}")
                return False
        
        # 检查轨迹文件
        if (hasattr(args, 'enable_geometric_constraints') and args.enable_geometric_constraints and
            hasattr(args, 'track_path') and args.track_path):
            if not os.path.exists(args.track_path):
                print(f"Warning: Track file not found: {args.track_path}")
                print("Geometric constraints will be disabled.")
                args.enable_geometric_constraints = False
        
        return True


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_constraint_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        加载约束配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典，如果加载失败返回None
        """
        if not os.path.exists(config_path):
            print(f"Warning: Constraint config file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Loaded constraint configuration from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading constraint config: {e}")
            return None
    
    @staticmethod
    def save_constraint_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        保存约束配置文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            
        Returns:
            保存是否成功
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Saved constraint configuration to: {config_path}")
            return True
        except Exception as e:
            print(f"Error saving constraint config: {e}")
            return False
    
    @staticmethod
    def create_default_constraint_config() -> Dict[str, Any]:
        """
        创建默认的约束配置
        
        Returns:
            默认配置字典
        """
        return {
            "weighting": {
                "texture_weight_min": 0.3,
                "texture_weight_max": 2.0,
                "texture_gradient_threshold": 100.0,
                "confidence_decay_factor": 2.0,
                "min_confidence_weight": 0.1,
                "early_stage_weight": 0.01,
                "mid_stage_weight": 0.05,
                "final_stage_weight": 0.2,
                "early_stage_iterations": 1000,
                "mid_stage_iterations": 5000
            },
            "quality": {
                "min_trajectory_length": 3,
                "min_quality_score": 0.4,
                "max_outlier_ratio": 0.3,
                "length_weight": 0.2,
                "visibility_weight": 0.3,
                "consistency_weight": 0.3,
                "stability_weight": 0.2,
                "outlier_threshold_pixels": 2.0,
                "ransac_threshold": 1.0,
                "min_inlier_ratio": 0.7
            },
            "multiscale": {
                "scales": [1.0, 0.5, 0.25],
                "scale_weights": [0.5, 0.3, 0.2],
                "consistency_threshold": 1.5,
                "consistency_weight": 0.1,
                "enable_scale_fallback": True,
                "min_valid_scales": 1
            },
            "validation": {
                "validation_interval": 100,
                "constraint_satisfaction_threshold": 0.85,
                "geometric_consistency_threshold": 0.9,
                "generate_detailed_reports": True,
                "save_validation_history": True,
                "max_history_length": 1000
            },
            "device": "cuda",
            "dtype": "float32",
            "enable_debug_mode": False,
            "batch_size": 1024,
            "num_workers": 4,
            "enable_parallel_processing": True
        }
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置字典
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def apply_config_to_args(args: Namespace, config: Dict[str, Any]):
        """
        将配置应用到命令行参数对象
        
        Args:
            args: 命令行参数对象
            config: 配置字典
        """
        # 映射配置到参数
        config_mapping = {
            # 权重配置
            'weighting.texture_weight_min': 'texture_weight_min',
            'weighting.texture_weight_max': 'texture_weight_max',
            'weighting.confidence_decay_factor': 'confidence_decay_factor',
            
            # 质量配置
            'quality.min_quality_score': 'min_trajectory_quality',
            'quality.max_outlier_ratio': 'max_outlier_ratio',
            'quality.outlier_threshold_pixels': 'outlier_threshold_pixels',
            
            # 多尺度配置
            'multiscale.scales': 'multiscale_scales',
            'multiscale.scale_weights': 'multiscale_weights',
            
            # 验证配置
            'validation.validation_interval': 'constraint_validation_interval',
            'validation.constraint_satisfaction_threshold': 'constraint_satisfaction_threshold',
        }
        
        for config_key, arg_name in config_mapping.items():
            keys = config_key.split('.')
            value = config
            
            try:
                for key in keys:
                    value = value[key]
                setattr(args, arg_name, value)
            except (KeyError, TypeError):
                continue  # 跳过不存在的配置项


def setup_geometric_constraints_config(args: Namespace) -> bool:
    """
    设置几何约束配置
    
    Args:
        args: 命令行参数对象
        
    Returns:
        设置是否成功
    """
    # 验证基础参数
    if not ConfigValidator.validate_file_paths(args):
        return False
    
    # 如果启用了几何约束
    if getattr(args, 'enable_geometric_constraints', False):
        # 加载配置文件
        config = None
        if hasattr(args, 'constraint_config_path') and args.constraint_config_path:
            config = ConfigManager.load_constraint_config(args.constraint_config_path)
        
        # 如果没有配置文件，使用默认配置
        if config is None:
            config = ConfigManager.create_default_constraint_config()
            print("Using default geometric constraints configuration")
        
        # 应用配置到参数
        ConfigManager.apply_config_to_args(args, config)
        
        # 验证几何约束配置
        if not ConfigValidator.validate_geometric_constraints_config(args):
            print("Geometric constraints configuration validation failed")
            return False
        
        print("Geometric constraints configuration validated successfully")
    
    return True


def print_geometric_constraints_summary(args: Namespace):
    """
    打印几何约束配置摘要
    
    Args:
        args: 命令行参数对象
    """
    if not getattr(args, 'enable_geometric_constraints', False):
        print("Geometric constraints: DISABLED")
        return
    
    print("=== Geometric Constraints Configuration ===")
    print(f"Track file: {getattr(args, 'track_path', 'Not specified')}")
    print(f"Constraint weight: {getattr(args, 'geometric_constraint_weight', 0.1)}")
    print(f"Multiscale weight: {getattr(args, 'multiscale_constraint_weight', 0.05)}")
    print(f"Min trajectory quality: {getattr(args, 'min_trajectory_quality', 0.4)}")
    print(f"Outlier threshold: {getattr(args, 'outlier_threshold_pixels', 2.0)} pixels")
    print(f"Robust loss type: {getattr(args, 'robust_loss_type', 'huber')}")
    print(f"Adaptive weighting: {'ENABLED' if getattr(args, 'enable_adaptive_weighting', True) else 'DISABLED'}")
    print("=" * 45)