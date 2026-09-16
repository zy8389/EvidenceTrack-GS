"""
约束系统配置管理

提供约束参数的配置、验证和管理功能
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import os


@dataclass
class WeightingConfig:
    """权重配置"""
    # 纹理权重参数
    texture_weight_min: float = 0.3
    texture_weight_max: float = 2.0
    texture_gradient_threshold: float = 100.0
    
    # 置信度权重参数
    confidence_decay_factor: float = 2.0
    min_confidence_weight: float = 0.1
    
    # 时序权重参数
    early_stage_weight: float = 0.01
    mid_stage_weight: float = 0.05
    final_stage_weight: float = 0.2
    early_stage_iterations: int = 1000
    mid_stage_iterations: int = 5000


@dataclass
class QualityConfig:
    """质量评估配置"""
    # 轨迹质量阈值
    min_trajectory_length: int = 3
    min_quality_score: float = 0.4
    max_outlier_ratio: float = 0.3
    
    # 质量评估权重
    length_weight: float = 0.2
    visibility_weight: float = 0.3
    consistency_weight: float = 0.3
    stability_weight: float = 0.2
    
    # 异常值检测参数
    outlier_threshold_pixels: float = 2.0
    ransac_threshold: float = 1.0
    min_inlier_ratio: float = 0.7


@dataclass
class MultiScaleConfig:
    """多尺度配置"""
    # 尺度设置
    scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    scale_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # 尺度一致性参数
    consistency_threshold: float = 1.5
    consistency_weight: float = 0.1
    
    # 降级策略
    enable_scale_fallback: bool = True
    min_valid_scales: int = 1


@dataclass
class ValidationConfig:
    """验证配置"""
    # 验证频率
    validation_interval: int = 100
    
    # 验证阈值
    constraint_satisfaction_threshold: float = 0.85
    geometric_consistency_threshold: float = 0.9
    
    # 报告设置
    generate_detailed_reports: bool = True
    save_validation_history: bool = True
    max_history_length: int = 1000


@dataclass
class ConstraintConfig:
    """约束系统主配置类"""
    
    # 子配置
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # 全局设置
    device: str = "cuda"
    dtype: str = "float32"
    enable_debug_mode: bool = False
    random_seed: Optional[int] = None
    
    # 性能设置
    batch_size: int = 1024
    num_workers: int = 4
    enable_parallel_processing: bool = True
    memory_limit_gb: Optional[float] = None
    
    def __post_init__(self):
        """配置验证"""
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        # 验证权重配置
        if self.weighting.texture_weight_min >= self.weighting.texture_weight_max:
            raise ValueError("texture_weight_min must be less than texture_weight_max")
        
        if not (0.0 <= self.weighting.min_confidence_weight <= 1.0):
            raise ValueError("min_confidence_weight must be between 0.0 and 1.0")
        
        # 验证质量配置
        if self.quality.min_trajectory_length < 2:
            raise ValueError("min_trajectory_length must be at least 2")
        
        if not (0.0 <= self.quality.min_quality_score <= 1.0):
            raise ValueError("min_quality_score must be between 0.0 and 1.0")
        
        # 验证多尺度配置
        if len(self.multiscale.scales) != len(self.multiscale.scale_weights):
            raise ValueError("scales and scale_weights must have the same length")
        
        if abs(sum(self.multiscale.scale_weights) - 1.0) > 1e-6:
            raise ValueError("scale_weights must sum to 1.0")
        
        # 验证设备设置
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        if self.dtype not in ["float32", "float64"]:
            raise ValueError("dtype must be 'float32' or 'float64'")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConstraintConfig':
        """从字典创建配置"""
        # 递归创建子配置
        weighting_config = WeightingConfig(**config_dict.get('weighting', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))
        multiscale_config = MultiScaleConfig(**config_dict.get('multiscale', {}))
        validation_config = ValidationConfig(**config_dict.get('validation', {}))
        
        # 创建主配置
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['weighting', 'quality', 'multiscale', 'validation']}
        
        return cls(
            weighting=weighting_config,
            quality=quality_config,
            multiscale=multiscale_config,
            validation=validation_config,
            **main_config
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ConstraintConfig':
        """从JSON文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'weighting': {
                'texture_weight_min': self.weighting.texture_weight_min,
                'texture_weight_max': self.weighting.texture_weight_max,
                'texture_gradient_threshold': self.weighting.texture_gradient_threshold,
                'confidence_decay_factor': self.weighting.confidence_decay_factor,
                'min_confidence_weight': self.weighting.min_confidence_weight,
                'early_stage_weight': self.weighting.early_stage_weight,
                'mid_stage_weight': self.weighting.mid_stage_weight,
                'final_stage_weight': self.weighting.final_stage_weight,
                'early_stage_iterations': self.weighting.early_stage_iterations,
                'mid_stage_iterations': self.weighting.mid_stage_iterations,
            },
            'quality': {
                'min_trajectory_length': self.quality.min_trajectory_length,
                'min_quality_score': self.quality.min_quality_score,
                'max_outlier_ratio': self.quality.max_outlier_ratio,
                'length_weight': self.quality.length_weight,
                'visibility_weight': self.quality.visibility_weight,
                'consistency_weight': self.quality.consistency_weight,
                'stability_weight': self.quality.stability_weight,
                'outlier_threshold_pixels': self.quality.outlier_threshold_pixels,
                'ransac_threshold': self.quality.ransac_threshold,
                'min_inlier_ratio': self.quality.min_inlier_ratio,
            },
            'multiscale': {
                'scales': self.multiscale.scales,
                'scale_weights': self.multiscale.scale_weights,
                'consistency_threshold': self.multiscale.consistency_threshold,
                'consistency_weight': self.multiscale.consistency_weight,
                'enable_scale_fallback': self.multiscale.enable_scale_fallback,
                'min_valid_scales': self.multiscale.min_valid_scales,
            },
            'validation': {
                'validation_interval': self.validation.validation_interval,
                'constraint_satisfaction_threshold': self.validation.constraint_satisfaction_threshold,
                'geometric_consistency_threshold': self.validation.geometric_consistency_threshold,
                'generate_detailed_reports': self.validation.generate_detailed_reports,
                'save_validation_history': self.validation.save_validation_history,
                'max_history_length': self.validation.max_history_length,
            },
            'device': self.device,
            'dtype': self.dtype,
            'enable_debug_mode': self.enable_debug_mode,
            'random_seed': self.random_seed,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'enable_parallel_processing': self.enable_parallel_processing,
            'memory_limit_gb': self.memory_limit_gb,
        }
    
    def save_to_file(self, config_path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in updates.items():
            if hasattr(self, key):
                if key in ['weighting', 'quality', 'multiscale', 'validation']:
                    # 更新子配置
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(self, key, value)
        
        # 重新验证配置
        self._validate_config()
    
    def get_torch_dtype(self):
        """获取PyTorch数据类型"""
        import torch
        if self.dtype == "float32":
            return torch.float32
        elif self.dtype == "float64":
            return torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
    
    def get_device(self):
        """获取PyTorch设备"""
        import torch
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")