import os
import sys
import argparse
import shutil
import numpy as np

def run_command(cmd, exit_on_fail=True):
    """
    执行一个shell命令，并检查其返回值。
    如果命令失败，可以选择退出脚本。
    """
    print(f"🚀 [RUNNING]: {cmd}")
    return_code = os.system(cmd)
    if return_code != 0:
        print(f"❌ [ERROR]: Command failed with exit code {return_code}")
        print(f"   > Failed command: {cmd}")
        if exit_on_fail:
            sys.exit(1)
    return return_code

def main():
    parser = argparse.ArgumentParser(
        description="Run a complete COLMAP pipeline on a subset of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source_path', required=True, type=str,
                        help="Path to the scene directory, e.g., '/workspace/360_v2/kitchen'.")
    parser.add_argument('--image_dir', type=str, default='images_8',
                        help="Name of the source image directory within the scene folder.")
    parser.add_argument('--n_views', type=int, default=24,
                        help="Number of views to select for the new reconstruction.")
    
    args = parser.parse_args()

    # --- 1. 设置路径和创建工作目录 ---
    full_image_source_dir = os.path.join(args.source_path, args.image_dir)
    if not os.path.isdir(full_image_source_dir):
        print(f"❌ [ERROR]: Source image directory not found at: {full_image_source_dir}")
        sys.exit(1)

    # 为本次运行创建一个干净的工作目录
    scene_name = os.path.basename(os.path.normpath(args.source_path))
    work_dir = f"colmap_{scene_name}_{args.n_views}views"
    
    if os.path.exists(work_dir):
        print(f"🧹 [INFO]: Removing old working directory: {work_dir}")
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    
    images_work_dir = os.path.join(work_dir, 'images')
    os.makedirs(images_work_dir)
    
    print(f"✅ [INFO]: Created new working directory at: {os.path.abspath(work_dir)}")

    # --- 2. 筛选并复制图像 ---
    all_images = sorted([f for f in os.listdir(full_image_source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(all_images) < args.n_views:
        print(f"⚠️ [WARNING]: Requested {args.n_views} views, but only {len(all_images)} available. Using all available images.")
        selected_images = all_images
    else:
        # 使用线性插值方法均匀地选取n_views张图片
        indices = np.linspace(0, len(all_images) - 1, args.n_views, dtype=int)
        selected_images = [all_images[i] for i in indices]

    print(f"📸 [INFO]: Selecting and copying {len(selected_images)} images:")
    for img_name in selected_images:
        shutil.copy(os.path.join(full_image_source_dir, img_name), os.path.join(images_work_dir, img_name))
    print(f"✅ [INFO]: All {len(selected_images)} images copied.")

    # --- 3. 运行标准的COLMAP稀疏重建流程 ---
    # 进入工作目录，所有COLMAP命令都在这里执行
    os.chdir(work_dir)
    db_path = "database.db"

    print("\n--- 🌟 Starting Sparse Reconstruction 🌟 ---")
    
    # 特征提取
    run_command(f"colmap feature_extractor --database_path {db_path} --image_path images")
    
    # 特征匹配
    run_command(f"colmap exhaustive_matcher --database_path {db_path}")

    # 稀疏重建 (建图)
    sparse_dir = "sparse"
    os.makedirs(sparse_dir)
    run_command(f"colmap mapper --database_path {db_path} --image_path images --output_path {sparse_dir}")

    # 检查稀疏重建是否成功
    if not os.listdir(sparse_dir):
        print("❌ [FATAL]: Sparse reconstruction failed to produce a model in the 'sparse' directory.")
        print("   > This can happen if images have insufficient overlap or lack distinct features.")
        sys.exit(1)

    print("✅ [SUCCESS]: Sparse reconstruction completed successfully!")
    print(f"   > New sparse model is in: {os.path.abspath(sparse_dir)}")

    # --- 4. (可选) 运行稠密重建流程 ---
    print("\n--- 🌟 Starting Dense Reconstruction 🌟 ---")
    dense_dir = "dense"
    
    # 图像去畸变并为稠密重建做准备
    run_command(f"colmap image_undistorter --image_path images --input_path {sparse_dir}/0 --output_path {dense_dir} --output_type COLMAP")
    
    # 稠密匹配
    run_command(f"colmap patch_match_stereo --workspace_path {dense_dir} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")
    
    # 稠密融合
    run_command(f"colmap stereo_fusion --workspace_path {dense_dir} --input_type geometric --output_path {dense_dir}/fused.ply")

    print("✅ [SUCCESS]: Dense reconstruction completed successfully!")
    print(f"   > Dense point cloud saved to: {os.path.abspath(os.path.join(dense_dir, 'fused.ply'))}")
    print("\n🎉 All processes finished! 🎉")


if __name__ == "__main__":
    main()