#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹可视化综合脚本
统一调用2D特征轨迹图和3D几何骨架图生成功能
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the repository root and this directory so both project packages and
# sibling visualization modules resolve when this file is run directly.
project_root = Path(__file__).resolve().parents[3]
visualization_dir = Path(__file__).resolve().parent
sys.path.extend([str(project_root), str(visualization_dir)])

from feature_trajectory_2d_visualizer import FeatureTrajectory2DVisualizer
from trajectory_3d_skeleton_visualizer import Trajectory3DSkeletonVisualizer


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_2d_visualization(images_path: str, tracks_h5: str, output_dir: str, max_images: int = 5):
    """运行2D特征轨迹可视化"""
    print("=" * 60)
    print("阶段一: 生成2D特征轨迹图")
    print("=" * 60)
    
    try:
        # 创建2D可视化器
        visualizer_2d = FeatureTrajectory2DVisualizer(images_path)
        
        # 加载图像信息
        print("正在加载图像信息...")
        visualizer_2d.load_images_info()
        
        # 加载轨迹数据
        print("正在加载轨迹数据...")
        success = visualizer_2d.load_tracks_data(tracks_h5)
        
        if not success:
            print("⚠️  轨迹数据加载失败，将使用测试数据")
        else:
            print(f"✅ 成功加载 {len(visualizer_2d.trajectories)} 条轨迹")
        
        # 创建可视化
        print("正在生成2D特征轨迹图...")
        output_2d = Path(output_dir) / "2d_trajectories"
        visualizer_2d.visualize_2d_trajectories(str(output_2d), max_images)
        
        print(f"✅ 2D特征轨迹图已保存到: {output_2d}")
        print(f"   - trajectory_2d_image_*.png: 单张图像的轨迹标注")
        print(f"   - trajectory_2d_summary.png: 轨迹统计汇总")
        
        return True
        
    except Exception as e:
        print(f"❌ 2D可视化失败: {e}")
        logging.error(f"2D visualization failed: {e}", exc_info=True)
        return False


def run_3d_visualization(tracks_h5: str, output_dir: str, views: list = None):
    """运行3D几何骨架图可视化"""
    print("=" * 60)
    print("阶段二: 生成3D几何骨架图")
    print("=" * 60)
    
    if views is None:
        views = ['oblique', 'front', 'side', 'top']
    
    try:
        # 创建3D可视化器
        visualizer_3d = Trajectory3DSkeletonVisualizer()
        
        # 加载3D数据
        print("正在加载3D轨迹数据...")
        success = visualizer_3d.load_tracks_3d_data(tracks_h5)
        
        if not success:
            print("⚠️  3D数据加载失败，将使用测试数据")
        else:
            print(f"✅ 成功加载 {len(visualizer_3d.points_3d)} 个3D点")
        
        # 生成可视化
        print(f"正在生成3D骨架图 ({', '.join(views)} 视角)...")
        output_3d = Path(output_dir) / "3d_skeleton"
        visualizer_3d.generate_3d_skeleton_visualizations(str(output_3d), views)
        
        print(f"✅ 3D几何骨架图已保存到: {output_3d}")
        print(f"   - trajectory_3d_skeleton_*.png: 各视角的3D骨架图")
        print(f"   - trajectory_3d_skeleton_multi_view.png: 多视角对比")
        print(f"   - trajectory_3d_skeleton_analysis.png: 统计分析")
        print(f"   - 3d_skeleton_info.json: 数据集信息")
        
        return True
        
    except Exception as e:
        print(f"❌ 3D可视化失败: {e}")
        logging.error(f"3D visualization failed: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="轨迹可视化综合工具 - 生成2D特征轨迹图和3D几何骨架图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 生成所有可视化 (有原始图片):
   python run_trajectory_visualization.py --images_path /path/to/images --tracks_h5 tracks.h5

2. 只使用虚拟图片进行演示:
   python run_trajectory_visualization.py --tracks_h5 tracks.h5

3. 只生成2D特征轨迹图:
   python run_trajectory_visualization.py --images_path /path/to/images --tracks_h5 tracks.h5 --only_2d

4. 只生成3D几何骨架图:
   python run_trajectory_visualization.py --tracks_h5 tracks.h5 --only_3d

5. 自定义3D视角:
   python run_trajectory_visualization.py --tracks_h5 tracks.h5 --views front side top oblique

注意:
- images_path 直接指向包含图片文件的目录
- tracks.h5 是COLMAP格式的轨迹文件
- 如果不指定images_path或路径不存在，会自动生成虚拟图片用于演示
- 支持的图片格式: jpg, jpeg, png, bmp, tiff
        """
    )
    
    parser.add_argument("--images_path", type=str,
                       help="图片目录路径 (直接指向包含图片的文件夹，如果不提供则使用虚拟图片)")
    parser.add_argument("--tracks_h5", type=str, required=True,
                       help="tracks.h5文件路径")
    parser.add_argument("--output_dir", type=str, default="./trajectory_visualizations",
                       help="输出目录 (默认: ./trajectory_visualizations)")
    
    # 控制选项
    parser.add_argument("--only_2d", action="store_true",
                       help="只生成2D特征轨迹图")
    parser.add_argument("--only_3d", action="store_true",
                       help="只生成3D几何骨架图")
    
    # 2D可视化选项
    parser.add_argument("--max_images", type=int, default=5,
                       help="最大处理图像数量 (默认: 5)")
    
    # 3D可视化选项
    parser.add_argument("--views", type=str, nargs='+',
                       choices=['front', 'side', 'top', 'oblique', 'back', 'bottom'],
                       default=['oblique', 'front', 'side', 'top'],
                       help="3D视角选择 (默认: oblique front side top)")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 验证参数
    if not Path(args.tracks_h5).exists():
        print(f"❌ 错误: tracks.h5文件不存在: {args.tracks_h5}")
        sys.exit(1)
    
    if args.only_2d and args.only_3d:
        print("❌ 错误: --only_2d 和 --only_3d 不能同时使用")
        sys.exit(1)
    
    # 检查图片路径（可选）
    if args.images_path and not Path(args.images_path).exists():
        print(f"⚠️  警告: 指定的图片路径不存在: {args.images_path}")
        print("将使用虚拟图片进行演示")
        args.images_path = None
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 轨迹可视化工具启动")
    print(f"📁 输出目录: {output_dir.absolute()}")
    print(f"📊 轨迹文件: {args.tracks_h5}")
    
    if args.images_path:
        print(f"🖼️  图片目录: {args.images_path}")
    else:
        print("🖼️  图片目录: 未指定 (将使用虚拟图片)")
    
    success_count = 0
    total_tasks = 0
    
    # 执行2D可视化
    if not args.only_3d:
        total_tasks += 1
        if run_2d_visualization(args.images_path, args.tracks_h5, str(output_dir), args.max_images):
            success_count += 1
        print()
    
    # 执行3D可视化
    if not args.only_2d:
        total_tasks += 1
        if run_3d_visualization(args.tracks_h5, str(output_dir), args.views):
            success_count += 1
        print()
    
    # 总结
    print("=" * 60)
    print("📋 任务完成总结")
    print("=" * 60)
    print(f"✅ 成功完成: {success_count}/{total_tasks} 个任务")
    
    if success_count == total_tasks:
        print("🎉 所有可视化任务完成成功!")
        print(f"📁 请查看输出目录: {output_dir.absolute()}")
        
        # 显示生成的文件
        print("\n📄 生成的文件:")
        for file_path in sorted(output_dir.rglob("*.png")):
            rel_path = file_path.relative_to(output_dir)
            print(f"   - {rel_path}")
        
        for file_path in sorted(output_dir.rglob("*.json")):
            rel_path = file_path.relative_to(output_dir)
            print(f"   - {rel_path}")
    else:
        print("⚠️  部分任务失败，请检查上述错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
