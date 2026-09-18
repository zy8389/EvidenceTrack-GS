import os
import sys
import shutil
import sqlite3
import numpy as np
import argparse

# --- 全局常量和版本检查 ---
IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

# --- 辅助函数 ---

def run_command(cmd, exit_on_fail=True):
    """
    执行一个shell命令，并检查其返回值。
    如果命令失败，可以选择退出脚本。
    """
    print(f"🚀 [RUNNING]: {cmd}")
    return_code = os.system(cmd)
    if return_code != 0:
        print(f"❌ [ERROR]: Command failed with exit code {return_code}")
        print(f"  > Failed command: {cmd}")
        if exit_on_fail:
            sys.exit(1)
    return return_code

def array_to_blob(array):
    """将Numpy数组转换为SQLite BLOB"""
    if IS_PYTHON3:
        return array.tobytes()
    else:
        # np.getbuffer is deprecated in Python 3
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    """将SQLite BLOB转换回Numpy数组"""
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def round_python3(number):
    """Python 3 的四舍五入行为，处理 .5 的情况"""
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

# --- COLMAP 数据库交互类 ---

class COLMAPDatabase(sqlite3.Connection):
    """
    一个封装了与COLMAP database.db文件交互的类。
    提供了创建表和基本操作的方法。
    """
    # SQL 创建表的语句
    CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL)"""
    CREATE_IMAGES_TABLE = f"""CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
        prior_tx REAL, prior_ty REAL, prior_tz REAL,
        CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""
    CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
    CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
    CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)"""
    CREATE_TWO_VIEW_GEOMETRIES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)"""
    CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

    CREATE_ALL = "; ".join([
        CREATE_CAMERAS_TABLE, CREATE_IMAGES_TABLE, CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE, CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE, CREATE_NAME_INDEX
    ])

    @staticmethod
    def connect(database_path):
        """静态方法，用于连接数据库并返回一个COLMAPDatabase实例"""
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        # 提供便捷方法来创建表
        self.create_tables = lambda: self.executescript(self.CREATE_ALL)

# --- 主流程函数 ---

def pipeline(scene_path, n_views, llffhold=8):
    """
    对一个已经完成COLMAP稀疏重建的场景，选择一个子集视图，
    并使用原始的相机内外参进行重新三角测量，最后生成稠密点云。

    Args:
        scene_path (str): 场景的绝对路径 (例如, '/workspace/llff/nerf_llff_data/flower')。
        n_views (int): 要选择用于新重建的视图数量。
        llffhold (int): LLFF数据集中用于划分训练/测试集的因子。
    """
    print(f"\n===== 🎬 Processing scene: {os.path.basename(scene_path)} with {n_views} views =====\n")

    # --- 1. 设置路径和创建工作目录 ---
    original_sparse_dir = os.path.join(scene_path, 'sparse', '0')
    original_images_dir = os.path.join(scene_path, 'images')

    if not os.path.isdir(original_sparse_dir):
        print(f"❌ [ERROR]: Original sparse directory not found at: {original_sparse_dir}")
        sys.exit(1)

    work_dir_name = f"{n_views}_views_reconstruction"
    work_dir = os.path.join(scene_path, work_dir_name)

    if os.path.exists(work_dir):
        print(f"🧹 [INFO]: Removing old working directory: {work_dir}")
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    print(f"✅ [INFO]: Created new working directory at: {work_dir}")

    # 创建子目录
    images_work_dir = os.path.join(work_dir, 'images')
    created_dir = os.path.join(work_dir, 'created') # 存放我们手动创建的相机和图像信息
    triangulated_dir = os.path.join(work_dir, 'triangulated') # 存放三角测量的结果
    dense_dir = os.path.join(work_dir, 'dense') # 存放稠密重建结果
    os.makedirs(images_work_dir)
    os.makedirs(created_dir)
    os.makedirs(triangulated_dir)

    db_path = os.path.join(work_dir, "database.db")

    # --- 2. 从原始模型中读取相机姿态并筛选图像 ---
    print("\n--- 📝 Reading original model and selecting images ---")
    
    # 为了读取文本文件，如果原始模型是二进制的，先转换
    original_images_txt = os.path.join(original_sparse_dir, 'images.txt')
    if not os.path.exists(original_images_txt):
        print("    > Converting original sparse model to TXT format...")
        run_command(f"colmap model_converter --input_path {original_sparse_dir} --output_path {original_sparse_dir} --output_type TXT")

    images_data = {}
    with open(original_images_txt, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[9]
                # 跳过注释行 (下一行是3D点)
                fid.readline()
                # 存储图像名到其姿态和相机ID的映射
                images_data[image_name] = elems[1:]

    all_images = sorted(images_data.keys())
    
    # 遵循LLFF数据集的划分方式，排除测试集
    train_img_list = [c for idx, c in enumerate(all_images) if idx % llffhold != 0]

    # 从训练集中均匀采样 n_views 个图像
    if n_views > 0 and n_views < len(train_img_list):
        indices = [int(round_python3(i)) for i in np.linspace(0, len(train_img_list) - 1, n_views)]
        selected_images = [train_img_list[i] for i in indices]
    else:
        selected_images = train_img_list

    print(f"📸 [INFO]: Selected {len(selected_images)} images for reconstruction.")
    for img_name in selected_images:
        shutil.copy(os.path.join(original_images_dir, img_name), os.path.join(images_work_dir, img_name))
    print(f"✅ [INFO]: All {len(selected_images)} images copied.")


    # --- 3. 为新的重建准备输入文件 ---
    print("\n--- 🛠️ Preparing input files for triangulation ---")
    # 拷贝相机内参文件，因为我们使用原始的相机模型
    shutil.copy(os.path.join(original_sparse_dir, 'cameras.txt'), created_dir)
    
    # 创建一个空的 3D 点文件，因为 `point_triangulator` 会自己生成
    with open(os.path.join(created_dir, 'points3D.txt'), "w") as fid:
        pass
        
    # --- 4. 在图像子集上运行特征提取和匹配 ---
    print("\n--- 🌟 Starting Feature Extraction & Matching 🌟 ---")
    run_command(f"colmap feature_extractor --database_path {db_path} --image_path {images_work_dir}")
    run_command(f"colmap exhaustive_matcher --database_path {db_path}")
    
    # --- 5. 创建新的 images.txt，使用原始姿态 ---
    print("\n--- ✍️  Writing new 'images.txt' with original poses ---")
    db = COLMAPDatabase.connect(db_path)
    # 按COLMAP内部处理的顺序获取图像名
    db_images = db.execute("SELECT name FROM images ORDER BY image_id")
    image_names_from_db = [row[0] for row in db_images]
    
    with open(os.path.join(created_dir, 'images.txt'), "w") as fid:
        for new_id, img_name in enumerate(image_names_from_db, 1):
            original_data = images_data[img_name]
            # 格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # 我们使用新的 new_id，但保留原始的姿态数据
            line_data = [str(new_id)] + original_data
            fid.write(" ".join(line_data) + "\n\n") # COLMAP需要一个空行
    print("✅ [INFO]: Successfully created 'images.txt' for the subset.")

    # --- 6. 运行点三角测量 ---
    print("\n--- 🌟 Starting Point Triangulation (using fixed poses) 🌟 ---")
    run_command(f"colmap point_triangulator "
                f"--database_path {db_path} "
                f"--image_path {images_work_dir} "
                f"--input_path {created_dir} "
                f"--output_path {triangulated_dir}")
    
    if not os.listdir(triangulated_dir):
        print("❌ [FATAL]: Point triangulation failed to produce a model.")
        sys.exit(1)
    
    print("✅ [SUCCESS]: Point triangulation completed.")
    
    # --- 7. 运行稠密重建 ---
    print("\n--- 🌟 Starting Dense Reconstruction 🌟 ---")
    run_command(f"colmap image_undistorter "
                f"--image_path {images_work_dir} "
                f"--input_path {triangulated_dir} "
                f"--output_path {dense_dir} "
                f"--output_type COLMAP")

    run_command(f"colmap patch_match_stereo "
                f"--workspace_path {dense_dir} "
                f"--workspace_format COLMAP "
                f"--PatchMatchStereo.geom_consistency true")
                
    run_command(f"colmap stereo_fusion "
                f"--workspace_path {dense_dir} "
                f"--input_type geometric "
                f"--output_path {os.path.join(dense_dir, 'fused.ply')}")

    print("✅ [SUCCESS]: Dense reconstruction completed successfully!")
    print(f"  > Dense point cloud saved to: {os.path.join(os.path.abspath(dense_dir), 'fused.ply')}")
    print("\n🎉 All processes finished! 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a COLMAP re-triangulation pipeline on a subset of images from an existing reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scene_path', required=True, type=str,
                        help="Absolute path to the scene directory (e.g., '/workspace/llff/nerf_llff_data/flower').")
    parser.add_argument('--n_views', required=True, type=int,
                        help="Number of views to select for the new reconstruction.")
    
    args = parser.parse_args()

    pipeline(scene_path=args.scene_path, n_views=args.n_views)
    
    # # ---- 或者，使用硬编码的路径进行测试 ----
    # base_path = '/workspace/llff/nerf_llff_data/' # 请确保这是绝对路径!
    # for scene in ['flower']:
    #     scene_full_path = os.path.join(base_path, scene)
    #     pipeline(scene_path=scene_full_path, n_views=20)