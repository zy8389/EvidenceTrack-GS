# -*- coding: utf-8 -*-
"""
实时几何约束验证系统

实现几何约束满足度计算、质量报告生成和参数自动校准功能
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 假设这些数据结构在其他地方定义
# from .data_structures import Trajectory, ConstraintResult, Point2D
# from .interfaces import ConstraintEngine

# --- 为使代码可独立运行，添加临时的模拟数据结构 ---
@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Trajectory:
    points: List[Point2D]
    quality_score: float = 1.0

    @property
    def length(self) -> int:
        return len(self.points)

    def get_points_tensor(self) -> torch.Tensor:
        return torch.tensor([[p.x, p.y] for p in self.points], dtype=torch.float32)

@dataclass
class ConstraintResult:
    mean_error: float = 0.0
    individual_errors: List[float] = None
    outlier_ratio: float = 0.0
    outlier_mask: torch.Tensor = None
    scale_contributions: Dict[str, float] = None

    def __post_init__(self):
        if self.individual_errors is None:
            self.individual_errors = []
        if self.outlier_mask is None:
            self.outlier_mask = torch.empty(0)
        if self.scale_contributions is None:
            self.scale_contributions = {}

class ConstraintEngine:
    """模拟的约束引擎接口"""
    def __init__(self):
        self.satisfaction_threshold = 0.85
        self.error_tolerance = 2.0
        self.outlier_threshold = 3.0
# --- 模拟数据结构结束 ---


@dataclass
class ValidationMetrics:
    """验证指标数据结构"""
    constraint_satisfaction: float  # 约束满足度 (0-1)
    geometric_consistency: float    # 几何一致性 (0-1)
    reprojection_accuracy: float    # 重投影精度
    outlier_ratio: float            # 异常值比例
    temporal_stability: float       # 时序稳定性
    scale_consistency: float        # 尺度一致性
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'constraint_satisfaction': self.constraint_satisfaction,
            'geometric_consistency': self.geometric_consistency,
            'reprojection_accuracy': self.reprojection_accuracy,
            'outlier_ratio': self.outlier_ratio,
            'temporal_stability': self.temporal_stability,
            'scale_consistency': self.scale_consistency
        }


@dataclass
class QualityReport:
    """几何质量报告数据结构"""
    timestamp: float
    iteration: int
    validation_metrics: ValidationMetrics
    problem_regions: List[Dict[str, Any]]
    quality_trends: Dict[str, List[float]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'iteration': self.iteration,
            'validation_metrics': self.validation_metrics.to_dict(),
            'problem_regions': self.problem_regions,
            'quality_trends': self.quality_trends,
            'recommendations': self.recommendations
        }


class GeometricConstraintSatisfactionCalculator:
    """几何约束满足度计算器"""
    
    def __init__(self, 
                 satisfaction_threshold: float = 0.85,
                 error_tolerance: float = 2.0,
                 consistency_window: int = 10):
        """
        初始化约束满足度计算器
        
        Args:
            satisfaction_threshold: 满足度阈值
            error_tolerance: 误差容忍度（像素）
            consistency_window: 一致性检查窗口大小
        """
        self.satisfaction_threshold = satisfaction_threshold
        self.error_tolerance = error_tolerance
        self.consistency_window = consistency_window
        
        # 历史数据存储
        self.error_history = deque(maxlen=1000)
        self.satisfaction_history = deque(maxlen=1000)
        self.outlier_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def compute_constraint_satisfaction(self, 
                                          constraint_result: ConstraintResult,
                                          trajectories: List[Trajectory]) -> ValidationMetrics:
        """
        计算约束满足度指标
        
        Args:
            constraint_result: 约束计算结果
            trajectories: 轨迹列表
            
        Returns:
            验证指标
        """
        # 1. 计算基础满足度
        satisfaction = self._compute_basic_satisfaction(constraint_result)
        
        # 2. 计算几何一致性
        consistency = self._compute_geometric_consistency(constraint_result, trajectories)
        
        # 3. 计算重投影精度
        accuracy = self._compute_reprojection_accuracy(constraint_result)
        
        # 4. 计算异常值比例
        outlier_ratio = constraint_result.outlier_ratio
        
        # 5. 计算时序稳定性
        temporal_stability = self._compute_temporal_stability(constraint_result)
        
        # 6. 计算尺度一致性
        scale_consistency = self._compute_scale_consistency(constraint_result)
        
        # 更新历史记录
        self._update_history(satisfaction, accuracy, outlier_ratio)
        
        return ValidationMetrics(
            constraint_satisfaction=satisfaction,
            geometric_consistency=consistency,
            reprojection_accuracy=accuracy,
            outlier_ratio=outlier_ratio,
            temporal_stability=temporal_stability,
            scale_consistency=scale_consistency
        )
    
    def _compute_basic_satisfaction(self, constraint_result: ConstraintResult) -> float:
        """计算基础约束满足度"""
        if not constraint_result.individual_errors:
            return 0.0
        
        # 计算满足误差阈值的约束比例
        satisfied_constraints = sum(1 for error in constraint_result.individual_errors 
                                      if error <= self.error_tolerance)
        total_constraints = len(constraint_result.individual_errors)
        
        satisfaction = satisfied_constraints / total_constraints if total_constraints > 0 else 0.0
        
        self.logger.debug(f"Basic satisfaction: {satisfaction:.3f} "
                          f"({satisfied_constraints}/{total_constraints})")
        
        return satisfaction
    
    def _compute_geometric_consistency(self, 
                                       constraint_result: ConstraintResult,
                                       trajectories: List[Trajectory]) -> float:
        """计算几何一致性"""
        if not trajectories:
            return 0.0
        
        consistency_scores = []
        
        for trajectory in trajectories:
            if trajectory.length < 3:
                continue
            
            # 计算轨迹内部的几何一致性
            points = trajectory.get_points_tensor()
            
            # 使用相邻点间距离的变化来衡量一致性
            distances = torch.norm(points[1:] - points[:-1], dim=1)
            if len(distances) > 1:
                distance_variance = torch.var(distances).item()
                # 将方差转换为一致性分数（方差越小，一致性越高）
                consistency = 1.0 / (1.0 + distance_variance / 100.0)
                consistency_scores.append(consistency)
        
        if not consistency_scores:
            return 0.0
        
        overall_consistency = sum(consistency_scores) / len(consistency_scores)
        
        self.logger.debug(f"Geometric consistency: {overall_consistency:.3f}")
        
        return overall_consistency
    
    def _compute_reprojection_accuracy(self, constraint_result: ConstraintResult) -> float:
        """计算重投影精度"""
        if not constraint_result.individual_errors:
            return 0.0
        
        # 使用平均误差的倒数作为精度指标
        mean_error = constraint_result.mean_error
        accuracy = 1.0 / (1.0 + mean_error)
        
        self.logger.debug(f"Reprojection accuracy: {accuracy:.3f} (mean error: {mean_error:.3f})")
        
        return accuracy
    
    def _compute_temporal_stability(self, constraint_result: ConstraintResult) -> float:
        """计算时序稳定性"""
        if len(self.error_history) < 2:
            return 1.0  # 初始状态认为是稳定的
        
        # 计算误差变化的稳定性
        recent_errors = list(self.error_history)[-self.consistency_window:]
        if len(recent_errors) < 2:
            return 1.0
        
        error_changes = [abs(recent_errors[i] - recent_errors[i-1]) 
                         for i in range(1, len(recent_errors))]
        
        if not error_changes:
            return 1.0
        
        mean_change = sum(error_changes) / len(error_changes)
        # 变化越小，稳定性越高
        stability = 1.0 / (1.0 + mean_change)
        
        self.logger.debug(f"Temporal stability: {stability:.3f}")
        
        return stability
    
    def _compute_scale_consistency(self, constraint_result: ConstraintResult) -> float:
        """计算尺度一致性"""
        if not constraint_result.scale_contributions:
            return 1.0  # 没有多尺度信息时认为一致
        
        # 计算不同尺度间贡献的一致性
        contributions = list(constraint_result.scale_contributions.values())
        if len(contributions) < 2:
            return 1.0
        
        # 使用标准差来衡量一致性
        mean_contrib = sum(contributions) / len(contributions)
        variance = sum((c - mean_contrib) ** 2 for c in contributions) / len(contributions)
        
        # 方差越小，一致性越高
        consistency = 1.0 / (1.0 + variance)
        
        self.logger.debug(f"Scale consistency: {consistency:.3f}")
        
        return consistency
    
    def _update_history(self, satisfaction: float, accuracy: float, outlier_ratio: float):
        """更新历史记录"""
        self.satisfaction_history.append(satisfaction)
        self.error_history.append(1.0 - accuracy)  # 将精度转换为误差
        self.outlier_history.append(outlier_ratio)
    
    def detect_constraint_failure(self, metrics: ValidationMetrics) -> Tuple[bool, List[str]]:
        """
        检测约束失效
        
        Args:
            metrics: 验证指标
            
        Returns:
            (是否失效, 失效原因列表)
        """
        failures = []
        
        # 检查约束满足度
        if metrics.constraint_satisfaction < self.satisfaction_threshold:
            failures.append(f"约束满足度过低: {metrics.constraint_satisfaction:.3f} < {self.satisfaction_threshold}")
        
        # 检查异常值比例
        if metrics.outlier_ratio > 0.3:
            failures.append(f"异常值比例过高: {metrics.outlier_ratio:.3f} > 0.3")
        
        # 检查几何一致性
        if metrics.geometric_consistency < 0.7:
            failures.append(f"几何一致性不足: {metrics.geometric_consistency:.3f} < 0.7")
        
        # 检查时序稳定性
        if metrics.temporal_stability < 0.6:
            failures.append(f"时序稳定性不足: {metrics.temporal_stability:.3f} < 0.6")
        
        is_failed = len(failures) > 0
        
        if is_failed:
            self.logger.warning(f"检测到约束失效: {'; '.join(failures)}")
        
        return is_failed, failures
    
    def get_quality_statistics(self) -> Dict[str, float]:
        """获取质量统计信息"""
        stats = {}
        
        if self.satisfaction_history:
            stats['satisfaction_mean'] = sum(self.satisfaction_history) / len(self.satisfaction_history)
            stats['satisfaction_std'] = np.std(list(self.satisfaction_history))
        
        if self.error_history:
            stats['error_mean'] = sum(self.error_history) / len(self.error_history)
            stats['error_std'] = np.std(list(self.error_history))
        
        if self.outlier_history:
            stats['outlier_mean'] = sum(self.outlier_history) / len(self.outlier_history)
            stats['outlier_std'] = np.std(list(self.outlier_history))
        
        return stats


class ReprojectionValidator:
    """实时重投影验证器 - 主要验证系统类"""
    
    def __init__(self, 
                 constraint_engine: ConstraintEngine,
                 validation_interval: int = 100,
                 report_dir: str = "validation_reports"):
        """
        初始化重投影验证器
        
        Args:
            constraint_engine: 约束引擎实例
            validation_interval: 验证间隔（迭代次数）
            report_dir: 报告输出目录
        """
        self.constraint_engine = constraint_engine
        self.validation_interval = validation_interval
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
        # 初始化子组件
        self.satisfaction_calculator = GeometricConstraintSatisfactionCalculator()
        
        # 验证历史记录
        self.validation_history = []
        self.failure_count = 0
        self.consecutive_failures = 0
        
        # 配置日志
        self.logger = logging.getLogger(__name__)
        
        # 质量趋势数据
        self.quality_trends = defaultdict(list)
    
    def validate_constraints(self, 
                             constraint_result: ConstraintResult,
                             trajectories: List[Trajectory],
                             iteration: int) -> Tuple[bool, ValidationMetrics]:
        """
        验证约束有效性
        
        Args:
            constraint_result: 约束计算结果
            trajectories: 轨迹列表
            iteration: 当前迭代次数
            
        Returns:
            (是否通过验证, 验证指标)
        """
        # 计算验证指标
        metrics = self.satisfaction_calculator.compute_constraint_satisfaction(
            constraint_result, trajectories)
        
        # 检测约束失效
        is_failed, failure_reasons = self.satisfaction_calculator.detect_constraint_failure(metrics)
        
        # 更新失效计数
        if is_failed:
            self.failure_count += 1
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # 记录验证历史
        self.validation_history.append({
            'iteration': iteration,
            'timestamp': time.time(),
            'metrics': metrics,
            'is_failed': is_failed,
            'failure_reasons': failure_reasons
        })
        
        # 更新质量趋势
        self._update_quality_trends(metrics)
        
        # 记录日志
        if is_failed:
            self.logger.warning(f"Iteration {iteration}: 约束验证失败 - {'; '.join(failure_reasons)}")
        else:
            self.logger.debug(f"Iteration {iteration}: 约束验证通过 - 满足度: {metrics.constraint_satisfaction:.3f}")
        
        return not is_failed, metrics
    
    def _update_quality_trends(self, metrics: ValidationMetrics):
        """更新质量趋势数据"""
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            self.quality_trends[key].append(value)
            
            # 保持趋势数据在合理范围内
            if len(self.quality_trends[key]) > 1000:
                self.quality_trends[key] = self.quality_trends[key][-1000:]
    
    def should_trigger_recalibration(self) -> bool:
        """
        判断是否应该触发参数重新校准
        
        Returns:
            是否需要重新校准
        """
        # 连续3次验证失败时触发重新校准
        return self.consecutive_failures >= 3
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证摘要信息"""
        if not self.validation_history:
            return {}
        
        total_validations = len(self.validation_history)
        failed_validations = sum(1 for v in self.validation_history if v['is_failed'])
        
        # 计算最近的指标
        recent_metrics = [v['metrics'] for v in self.validation_history[-10:]]
        if recent_metrics:
            avg_satisfaction = sum(m.constraint_satisfaction for m in recent_metrics) / len(recent_metrics)
            avg_consistency = sum(m.geometric_consistency for m in recent_metrics) / len(recent_metrics)
        else:
            avg_satisfaction = 0.0
            avg_consistency = 0.0
        
        return {
            'total_validations': total_validations,
            'failed_validations': failed_validations,
            'failure_rate': failed_validations / total_validations if total_validations > 0 else 0.0,
            'consecutive_failures': self.consecutive_failures,
            'recent_avg_satisfaction': avg_satisfaction,
            'recent_avg_consistency': avg_consistency,
            'quality_statistics': self.satisfaction_calculator.get_quality_statistics()
        }

    def generate_quality_report(self, 
                                  constraint_result: ConstraintResult,
                                  trajectories: List[Trajectory],
                                  iteration: int) -> QualityReport:
        """
        生成几何质量报告
        
        Args:
            constraint_result: 约束计算结果
            trajectories: 轨迹列表
            iteration: 当前迭代次数
            
        Returns:
            质量报告
        """
        # 计算验证指标
        metrics = self.satisfaction_calculator.compute_constraint_satisfaction(
            constraint_result, trajectories)
        
        # 识别问题区域
        problem_regions = self._identify_problem_regions(constraint_result, trajectories)
        
        # 生成改进建议
        recommendations = self._generate_recommendations(metrics, problem_regions)
        
        # 创建质量报告
        report = QualityReport(
            timestamp=time.time(),
            iteration=iteration,
            validation_metrics=metrics,
            problem_regions=problem_regions,
            quality_trends=dict(self.quality_trends),
            recommendations=recommendations
        )
        
        # 保存报告
        self._save_quality_report(report)
        
        return report
    
    def _identify_problem_regions(self, 
                                  constraint_result: ConstraintResult,
                                  trajectories: List[Trajectory]) -> List[Dict[str, Any]]:
        """识别问题区域"""
        problem_regions = []
        
        # 1. 识别高误差轨迹
        if constraint_result.individual_errors:
            error_threshold = np.percentile(constraint_result.individual_errors, 90)
            high_error_indices = [i for i, error in enumerate(constraint_result.individual_errors) 
                                  if error > error_threshold]
            
            if high_error_indices:
                problem_regions.append({
                    'type': 'high_error_trajectories',
                    'description': f'发现 {len(high_error_indices)} 条高误差轨迹',
                    'trajectory_indices': high_error_indices,
                    'error_threshold': error_threshold,
                    'severity': 'high' if len(high_error_indices) > len(trajectories) * 0.2 else 'medium'
                })
        
        # 2. 识别异常值聚集区域
        if constraint_result.outlier_mask.sum() > 0:
            outlier_indices = torch.where(constraint_result.outlier_mask)[0].tolist()
            problem_regions.append({
                'type': 'outlier_clusters',
                'description': f'检测到 {len(outlier_indices)} 个异常值',
                'outlier_indices': outlier_indices,
                'outlier_ratio': constraint_result.outlier_ratio,
                'severity': 'high' if constraint_result.outlier_ratio > 0.3 else 'medium'
            })
        
        # 3. 识别低质量轨迹
        low_quality_trajectories = [i for i, traj in enumerate(trajectories) 
                                    if traj.quality_score < 0.4]
        if low_quality_trajectories:
            problem_regions.append({
                'type': 'low_quality_trajectories',
                'description': f'发现 {len(low_quality_trajectories)} 条低质量轨迹',
                'trajectory_indices': low_quality_trajectories,
                'quality_threshold': 0.4,
                'severity': 'medium'
            })
        
        # 4. 识别不稳定区域（基于历史数据）
        if len(self.validation_history) >= 5:
            recent_satisfactions = [v['metrics'].constraint_satisfaction 
                                    for v in self.validation_history[-5:]]
            satisfaction_variance = np.var(recent_satisfactions)
            
            if satisfaction_variance > 0.01:  # 满足度变化较大
                problem_regions.append({
                    'type': 'unstable_region',
                    'description': '约束满足度存在较大波动',
                    'variance': satisfaction_variance,
                    'recent_values': recent_satisfactions,
                    'severity': 'medium'
                })
        
        return problem_regions
    
    def _generate_recommendations(self, 
                                  metrics: ValidationMetrics,
                                  problem_regions: List[Dict[str, Any]]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于整体指标的建议
        if metrics.constraint_satisfaction < 0.7:
            recommendations.append("约束满足度较低，建议调整约束参数或增加训练迭代")
        
        if metrics.outlier_ratio > 0.25:
            recommendations.append("异常值比例过高，建议检查轨迹质量或调整异常值检测阈值")
        
        if metrics.geometric_consistency < 0.6:
            recommendations.append("几何一致性不足，建议增强轨迹预处理或调整一致性约束权重")
        
        if metrics.temporal_stability < 0.5:
            recommendations.append("时序稳定性差，建议降低学习率或增加正则化")
        
        # 基于问题区域的建议
        for region in problem_regions:
            if region['type'] == 'high_error_trajectories' and region['severity'] == 'high':
                recommendations.append("存在大量高误差轨迹，建议重新评估轨迹质量或调整权重分配")
            
            elif region['type'] == 'outlier_clusters' and region['severity'] == 'high':
                recommendations.append("异常值聚集严重，建议检查数据质量或增强异常值处理")
            
            elif region['type'] == 'unstable_region':
                recommendations.append("约束满足度波动较大，建议调整优化器参数或增加训练稳定性")
        
        # 性能优化建议
        if len(self.validation_history) > 0:
            recent_failure_rate = sum(1 for v in self.validation_history[-10:] if v['is_failed']) / min(10, len(self.validation_history))
            if recent_failure_rate > 0.3:
                recommendations.append("近期验证失败率较高，建议进行参数重新校准")
        
        return recommendations
    
    def _save_quality_report(self, report: QualityReport):
        """保存质量报告"""
        # 保存JSON格式报告
        report_file = self.report_dir / f"quality_report_iter_{report.iteration}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 生成可视化报告
        self._generate_visual_report(report)
        
        self.logger.info(f"质量报告已保存: {report_file}")
    
    def _generate_visual_report(self, report: QualityReport):
        """生成可视化报告"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'几何质量报告 - 迭代 {report.iteration}', fontsize=16)
            
            # 1. 验证指标雷达图
            self._plot_metrics_radar(axes[0, 0], report.validation_metrics)
            
            # 2. 质量趋势图
            self._plot_quality_trends(axes[0, 1], report.quality_trends)
            
            # 3. 问题区域分布
            self._plot_problem_regions(axes[1, 0], report.problem_regions)
            
            # 4. 历史验证结果
            self._plot_validation_history(axes[1, 1])
            
            # 保存图表
            chart_file = self.report_dir / f"quality_chart_iter_{report.iteration}.png"
            plt.tight_layout()
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"可视化报告已保存: {chart_file}")
            
        except Exception as e:
            self.logger.warning(f"生成可视化报告失败: {e}")
    
    def _plot_metrics_radar(self, ax, metrics: ValidationMetrics):
        """绘制指标雷达图"""
        categories = ['约束满足度', '几何一致性', '重投影精度', '时序稳定性', '尺度一致性']
        values = [
            metrics.constraint_satisfaction,
            metrics.geometric_consistency,
            metrics.reprojection_accuracy,
            metrics.temporal_stability,
            metrics.scale_consistency
        ]
        
        # 添加第一个点到末尾以闭合雷达图
        values += values[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label='当前指标')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('验证指标概览')
        ax.grid(True)
    
    def _plot_quality_trends(self, ax, quality_trends: Dict[str, List[float]]):
        """绘制质量趋势图"""
        if not quality_trends:
            ax.text(0.5, 0.5, '暂无趋势数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('质量趋势')
            return
        
        for metric_name, values in quality_trends.items():
            if len(values) > 1:
                ax.plot(values[-50:], label=metric_name, alpha=0.7)  # 只显示最近50个点
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('指标值')
        ax.set_title('质量趋势 (最近50次)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_problem_regions(self, ax, problem_regions: List[Dict[str, Any]]):
        """绘制问题区域分布"""
        if not problem_regions:
            ax.text(0.5, 0.5, '未发现问题区域', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('问题区域分析')
            return
        
        # 统计问题类型
        problem_types = {}
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for region in problem_regions:
            problem_type = region['type']
            severity = region.get('severity', 'medium')
            
            problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
            severity_counts[severity] += 1
        
        # 绘制问题类型分布
        if problem_types:
            ax.pie(problem_types.values(), labels=problem_types.keys(), autopct='%1.1f%%')
            ax.set_title('问题区域类型分布')
    
    def _plot_validation_history(self, ax):
        """绘制验证历史"""
        if not self.validation_history:
            ax.text(0.5, 0.5, '暂无历史数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('验证历史')
            return
        
        # 提取最近的验证结果
        recent_history = self.validation_history[-100:]  # 最近100次
        iterations = [v['iteration'] for v in recent_history]
        satisfactions = [v['metrics'].constraint_satisfaction for v in recent_history]
        failures = [1 if v['is_failed'] else 0 for v in recent_history]
        
        # 绘制满足度趋势
        ax2 = ax.twinx()
        ax.plot(iterations, satisfactions, 'b-', label='约束满足度', alpha=0.7)
        ax2.bar(iterations, failures, alpha=0.3, color='red', label='验证失败')
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('约束满足度', color='blue')
        ax2.set_ylabel('验证失败', color='red')
        ax.set_title('验证历史 (最近100次)')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')


class ConstraintParameterCalibrator:
    """约束参数自动校准器"""
    
    def __init__(self, 
                 calibration_threshold: int = 5,
                 parameter_bounds: Dict[str, Tuple[float, float]] = None):
        """
        初始化参数校准器
        
        Args:
            calibration_threshold: 触发校准的连续失败次数
            parameter_bounds: 参数调整范围
        """
        self.calibration_threshold = calibration_threshold
        self.parameter_bounds = parameter_bounds or {
            'satisfaction_threshold': (0.6, 0.95),
            'error_tolerance': (1.0, 5.0),
            'outlier_threshold': (2.0, 4.0)
        }
        
        self.calibration_history = []
        self.logger = logging.getLogger(__name__)
    
    def should_calibrate(self, consecutive_failures: int, validation_metrics: ValidationMetrics) -> bool:
        """
        判断是否需要进行参数校准
        
        Args:
            consecutive_failures: 连续失败次数
            validation_metrics: 验证指标
            
        Returns:
            是否需要校准
        """
        # 基于连续失败次数
        if consecutive_failures >= self.calibration_threshold:
            return True
        
        # 基于指标质量
        if (validation_metrics.constraint_satisfaction < 0.5 or 
            validation_metrics.outlier_ratio > 0.4):
            return True
        
        return False
    
    def calibrate_parameters(self, 
                             constraint_engine: ConstraintEngine,
                             recent_metrics: List[ValidationMetrics]) -> Dict[str, float]:
        """
        执行参数校准
        
        Args:
            constraint_engine: 约束引擎
            recent_metrics: 最近的验证指标
            
        Returns:
            校准后的参数
        """
        if not recent_metrics:
            return {}
        
        # 分析当前问题
        avg_satisfaction = sum(m.constraint_satisfaction for m in recent_metrics) / len(recent_metrics)
        avg_outlier_ratio = sum(m.outlier_ratio for m in recent_metrics) / len(recent_metrics)
        
        new_parameters = {}
        
        # 调整满足度阈值
        if avg_satisfaction < 0.7:
            # 降低满足度阈值
            current_threshold = getattr(constraint_engine, 'satisfaction_threshold', 0.85)
            new_threshold = max(current_threshold * 0.9, self.parameter_bounds['satisfaction_threshold'][0])
            new_parameters['satisfaction_threshold'] = new_threshold
            self.logger.info(f"调整满足度阈值: {current_threshold:.3f} -> {new_threshold:.3f}")
        
        # 调整误差容忍度
        if avg_outlier_ratio > 0.3:
            # 增加误差容忍度
            current_tolerance = getattr(constraint_engine, 'error_tolerance', 2.0)
            new_tolerance = min(current_tolerance * 1.2, self.parameter_bounds['error_tolerance'][1])
            new_parameters['error_tolerance'] = new_tolerance
            self.logger.info(f"调整误差容忍度: {current_tolerance:.3f} -> {new_tolerance:.3f}")
        
        # 记录校准历史
        self.calibration_history.append({
            'timestamp': time.time(),
            'trigger_metrics': {
                'avg_satisfaction': avg_satisfaction,
                'avg_outlier_ratio': avg_outlier_ratio
            },
            'parameter_changes': new_parameters
        })
        
        # 应用新参数
        self._apply_parameters(constraint_engine, new_parameters)
        
        return new_parameters
    
    def _apply_parameters(self, constraint_engine: ConstraintEngine, parameters: Dict[str, float]):
        """应用新参数到约束引擎"""
        for param_name, param_value in parameters.items():
            if hasattr(constraint_engine, param_name):
                setattr(constraint_engine, param_name, param_value)
                self.logger.debug(f"应用参数 {param_name} = {param_value}")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """获取校准摘要"""
        return {
            'total_calibrations': len(self.calibration_history),
            'recent_calibrations': self.calibration_history[-5:] if self.calibration_history else [],
            'parameter_bounds': self.parameter_bounds
        }


class EnhancedReprojectionValidator(ReprojectionValidator):
    """增强的重投影验证器，集成自动校准功能"""
    
    def __init__(self, 
                 constraint_engine: ConstraintEngine,
                 validation_interval: int = 100,
                 report_dir: str = "validation_reports",
                 enable_auto_calibration: bool = True):
        """
        初始化增强的重投影验证器
        
        Args:
            constraint_engine: 约束引擎实例
            validation_interval: 验证间隔（迭代次数）
            report_dir: 报告输出目录
            enable_auto_calibration: 是否启用自动校准
        """
        super().__init__(constraint_engine, validation_interval, report_dir)
        
        self.enable_auto_calibration = enable_auto_calibration
        self.calibrator = ConstraintParameterCalibrator() if enable_auto_calibration else None
        
        # 校准相关状态
        self.last_calibration_iteration = 0
        self.calibration_cooldown = 500  # 校准冷却期（迭代次数）
    
    def validate_and_calibrate(self, 
                               constraint_result: ConstraintResult,
                               trajectories: List[Trajectory],
                               iteration: int) -> Tuple[bool, ValidationMetrics, Optional[Dict[str, float]]]:
        """
        验证约束并在需要时进行校准
        
        Args:
            constraint_result: 约束计算结果
            trajectories: 轨迹列表
            iteration: 当前迭代次数
            
        Returns:
            (是否通过验证, 验证指标, 校准参数变化)
        """
        # 执行常规验证
        is_valid, metrics = self.validate_constraints(constraint_result, trajectories, iteration)
        
        calibration_changes = None
        
        # 检查是否需要校准
        if (self.enable_auto_calibration and 
            self.calibrator and 
            iteration - self.last_calibration_iteration > self.calibration_cooldown):
            
            if self.calibrator.should_calibrate(self.consecutive_failures, metrics):
                # 获取最近的验证指标
                recent_metrics = [v['metrics'] for v in self.validation_history[-10:]]
                
                # 执行校准
                calibration_changes = self.calibrator.calibrate_parameters(
                    self.constraint_engine, recent_metrics)
                
                if calibration_changes:
                    self.last_calibration_iteration = iteration
                    self.consecutive_failures = 0  # 重置失败计数
                    
                    self.logger.info(f"Iteration {iteration}: 执行参数校准 - {calibration_changes}")
        
        return is_valid, metrics, calibration_changes
    
    def generate_comprehensive_report(self, 
                                      constraint_result: ConstraintResult,
                                      trajectories: List[Trajectory],
                                      iteration: int) -> Dict[str, Any]:
        """
        生成综合报告，包含验证、质量分析和校准信息
        
        Args:
            constraint_result: 约束计算结果
            trajectories: 轨迹列表
            iteration: 当前迭代次数
            
        Returns:
            综合报告字典
        """
        # 生成质量报告
        quality_report = self.generate_quality_report(constraint_result, trajectories, iteration)
        
        # 获取验证摘要
        validation_summary = self.get_validation_summary()
        
        # 获取校准摘要
        calibration_summary = (self.calibrator.get_calibration_summary() 
                               if self.calibrator else {})
        
        # 组合综合报告
        comprehensive_report = {
            'iteration': iteration,
            'timestamp': time.time(),
            'quality_report': quality_report.to_dict(),
            'validation_summary': validation_summary,
            'calibration_summary': calibration_summary,
            'system_status': {
                'auto_calibration_enabled': self.enable_auto_calibration,
                'consecutive_failures': self.consecutive_failures,
                'last_calibration_iteration': self.last_calibration_iteration,
                'validation_interval': self.validation_interval
            }
        }
        
        # 保存综合报告
        report_file = self.report_dir / f"comprehensive_report_iter_{iteration}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"综合报告已保存: {report_file}")
        
        return comprehensive_report
    
    def export_validation_data(self, output_file: str):
        """
        导出验证数据用于分析
        
        Args:
            output_file: 输出文件路径
        """
        export_data = {
            'validation_history': self.validation_history,
            'quality_trends': dict(self.quality_trends),
            'calibration_history': (self.calibrator.calibration_history 
                                    if self.calibrator else []),
            'statistics': {
                'total_validations': len(self.validation_history),
                'failure_count': self.failure_count,
                'consecutive_failures': self.consecutive_failures,
                'quality_statistics': self.satisfaction_calculator.get_quality_statistics()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"验证数据已导出: {output_file}")
    
    def reset_validation_state(self):
        """重置验证状态"""
        self.validation_history.clear()
        self.failure_count = 0
        self.consecutive_failures = 0
        self.quality_trends.clear()
        
        if self.calibrator:
            self.calibrator.calibration_history.clear()
        
        self.satisfaction_calculator.error_history.clear()
        self.satisfaction_calculator.satisfaction_history.clear()
        self.satisfaction_calculator.outlier_history.clear()
        
        self.logger.info("验证状态已重置")


# 工厂函数
def create_reprojection_validator(constraint_engine: ConstraintEngine,
                                  config: Dict[str, Any] = None) -> EnhancedReprojectionValidator:
    """
    创建重投影验证器的工厂函数
    
    Args:
        constraint_engine: 约束引擎实例
        config: 配置参数
        
    Returns:
        配置好的验证器实例
    """
    config = config or {}
    
    return EnhancedReprojectionValidator(
        constraint_engine=constraint_engine,
        validation_interval=config.get('validation_interval', 100),
        report_dir=config.get('report_dir', 'validation_reports'),
        enable_auto_calibration=config.get('enable_auto_calibration', True)
    )


# 辅助函数
def analyze_validation_trends(validation_data_file: str) -> Dict[str, Any]:
    """
    分析验证趋势数据
    
    Args:
        validation_data_file: 验证数据文件路径
        
    Returns:
        趋势分析结果
    """
    try:
        with open(validation_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        validation_history = data.get('validation_history', [])
        quality_trends = data.get('quality_trends', {})
        
        if not validation_history:
            return {'error': '没有验证历史数据'}
        
        # 计算趋势统计
        satisfactions = [v['metrics']['constraint_satisfaction'] for v in validation_history 
                         if 'metrics' in v and v['metrics'] is not None]
        failures = [v['is_failed'] for v in validation_history]
        
        analysis = {
            'total_validations': len(validation_history),
            'failure_rate': sum(failures) / len(failures) if failures else 0,
            'satisfaction_stats': {
                'mean': np.mean(satisfactions) if satisfactions else 0,
                'std': np.std(satisfactions) if satisfactions else 0,
                'min': min(satisfactions) if satisfactions else 0,
                'max': max(satisfactions) if satisfactions else 0
            },
            'trend_analysis': {}
        }
        
        # 分析各指标趋势
        for metric_name, values in quality_trends.items():
            if len(values) > 10:
                # 计算趋势斜率（简单线性回归）
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                analysis['trend_analysis'][metric_name] = {
                    'slope': slope,
                    'trend': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
                    'recent_mean': np.mean(values[-10:]),
                    'overall_mean': np.mean(values)
                }
        
        return analysis
        
    except Exception as e:
        return {'error': f'分析失败: {str(e)}'}


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 创建一个模拟的约束引擎
    mock_constraint_engine = ConstraintEngine()
    
    # 2. 使用工厂函数创建验证器
    validator = create_reprojection_validator(mock_constraint_engine)
    
    # 3. 模拟多次迭代和验证
    for i in range(1, 201):
        # 模拟生成约束结果和轨迹
        mock_error = max(0.1, 5.0 - i * 0.02 + np.random.randn() * 0.5)
        mock_outlier_ratio = max(0.0, 0.4 - i * 0.002 + np.random.randn() * 0.05)
        
        mock_result = ConstraintResult(
            mean_error=mock_error,
            individual_errors=list(np.random.rand(20) * mock_error * 2),
            outlier_ratio=mock_outlier_ratio,
            outlier_mask=torch.rand(20) < mock_outlier_ratio
        )
        mock_trajectories = [Trajectory(points=[Point2D(x,y) for x,y in np.random.rand(5,2)]) for _ in range(10)]
        
        # 执行验证和校准
        is_valid, metrics, changes = validator.validate_and_calibrate(mock_result, mock_trajectories, i)
        
        # 每50次迭代生成一次综合报告
        if i % 50 == 0:
            logging.info(f"--- 生成第 {i} 次迭代的综合报告 ---")
            validator.generate_comprehensive_report(mock_result, mock_trajectories, i)

    logging.info("模拟运行完成。报告已生成在 'validation_reports' 文件夹中。")
