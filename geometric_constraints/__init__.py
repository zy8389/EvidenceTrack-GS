"""GeoTrack geometric constraints with lazy legacy imports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "Trajectory": ("data_structures", "Trajectory"),
    "Point2D": ("data_structures", "Point2D"),
    "ConstraintResult": ("data_structures", "ConstraintResult"),
    "ConstraintEngine": ("interfaces", "ConstraintEngine"),
    "QualityAssessor": ("interfaces", "QualityAssessor"),
    "TrajectoryManager": ("interfaces", "TrajectoryManager"),
    "ConstraintConfig": ("config", "ConstraintConfig"),
    "TrajectoryManagerImpl": ("trajectory_manager", "TrajectoryManagerImpl"),
    "QualityAssessorImpl": ("trajectory_manager", "QualityAssessorImpl"),
    "MultiScaleImagePyramid": ("multiscale_constraints", "MultiScaleImagePyramid"),
    "MultiScaleConstraints": ("multiscale_constraints", "MultiScaleConstraints"),
    "MultiScaleConstraintEngine": ("multiscale_constraints", "MultiScaleConstraintEngine"),
    "ScaleConsistencyConstraint": ("multiscale_constraints", "ScaleConsistencyConstraint"),
    "EnhancedMultiScaleConstraintEngine": ("multiscale_constraints", "EnhancedMultiScaleConstraintEngine"),
    "EnhancedReprojectionConstraint": ("constraint_engine", "EnhancedReprojectionConstraint"),
    "ConstraintFusion": ("constraint_engine", "ConstraintFusion"),
    "GeometricConsistencyChecker": ("constraint_engine", "GeometricConsistencyChecker"),
    "ConstraintEngineImpl": ("constraint_engine", "ConstraintEngineImpl"),
    "AdvancedOutlierDetector": ("constraint_engine", "AdvancedOutlierDetector"),
    "EnhancedConstraintEngine": ("constraint_engine", "EnhancedConstraintEngine"),
    "HuberLoss": ("constraint_engine", "HuberLoss"),
    "RobustEstimator": ("constraint_engine", "RobustEstimator"),
    "ReprojectionStats": ("constraint_engine", "ReprojectionStats"),
    "ValidationMetrics": ("reprojection_validator", "ValidationMetrics"),
    "QualityReport": ("reprojection_validator", "QualityReport"),
    "GeometricConstraintSatisfactionCalculator": ("reprojection_validator", "GeometricConstraintSatisfactionCalculator"),
    "ReprojectionValidator": ("reprojection_validator", "ReprojectionValidator"),
    "ConstraintParameterCalibrator": ("reprojection_validator", "ConstraintParameterCalibrator"),
    "EnhancedReprojectionValidator": ("reprojection_validator", "EnhancedReprojectionValidator"),
    "create_reprojection_validator": ("reprojection_validator", "create_reprojection_validator"),
    "analyze_validation_trends": ("reprojection_validator", "analyze_validation_trends"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
