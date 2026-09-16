# GeoTrack-GS

本项目利用 3D 高斯溅射（3D Gaussian Splatting）技术，结合先进的几何约束系统和GT-DCA增强外观建模，实现高质量的 3D 场景重建与渲染。该项目基于论文《3D Gaussian Splatting for Real-Time Radiance Field Rendering》，并集成了多尺度几何约束、自适应权重调整和GT-DCA外观增强等创新功能。

## 📚 目录 (Table of Contents)

- [🚀 5分钟快速开始](#-5分钟快速开始)
- [主要功能 (Features)](#主要功能-features)
  - [🎯 核心技术特性](#-核心技术特性)
  - [🚀 GT-DCA 增强外观建模](#-gt-dca-增强外观建模)
  - [🔧 几何约束系统](#-几何约束系统)
  - [🎯 几何先验各向异性正则化](#-几何先验各向异性正则化-new)
  - [🛠️ 工程特性](#️-工程特性)
- [先决条件 (Prerequisites)](#先决条件-prerequisites)
- [安装 (Installation)](#安装-installation)
  - [🏗️ COLMAP 源码编译安装](#️-colmap-源码编译安装-linux服务器)
  - [🚀 快速安装](#-快速安装)
  - [🔧 高级安装选项](#-高级安装选项)
- [使用方法 (Usage)](#使用方法-usage)
  - [1. 准备数据](#1-准备数据)
  - [2. 训练](#2-训练)
  - [3. 渲染](#3-渲染)
  - [4. 评估](#4-评估)
  - [5. 配置文件](#5-配置文件)
  - [6. GT-DCA 详细说明](#6-gt-dca-详细说明)
  - [7. 几何先验各向异性正则化详解](#7-几何先验各向异性正则化详解-new)
- [❓ 常见问题 (FAQ)](#-常见问题-faq)
- [如何贡献 (Contributing)](#如何贡献-contributing)
- [致谢 (Acknowledgements)](#致谢-acknowledgements)
- [许可证 (License)](#许可证-license)

## 🚀 5分钟快速开始

想要快速体验GeoTrack-GS？按照以下步骤即可在5分钟内开始训练你的第一个模型：

### 📋 前提条件检查
```bash
# 检查Python版本 (需要3.8+)
python --version

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查GPU显存 (推荐8GB+)
nvidia-smi
```

### ⚡ 一键安装
```bash
# 1. 克隆项目 (包含所有子模块)
git clone --recursive https://github.com/CPy255/GeoTrack-GS.git
cd GeoTrack-GS

# 2. 创建环境并安装依赖
conda env create -f environment.yml
conda activate geotrack

# 3. 安装PyTorch (自动选择CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 构建扩展模块
cd submodules/diff-gaussian-rasterization-confidence && pip install . && cd ../..
cd submodules/simple-knn && pip install . && cd ../..

# 5. 验证安装
python debug/test_modules.py
```

### 🎯 快速训练示例
```bash
# 下载示例数据集 (可选)
git clone https://github.com/graphdeco-inria/gaussian-splatting-data.git data

# 基础训练 (约10-15分钟)
python train.py -s data/tandt/train -m output/quick_test --iterations 7000

# GT-DCA增强训练 (约15-20分钟)
python train.py -s data/tandt/train -m output/gtdca_test \
    --use_gt_dca \
    --gt_dca_feature_dim 128 \
    --gt_dca_num_sample_points 4 \
    --iterations 7000

# 查看训练结果
python render.py -m output/quick_test
python full_eval.py -m output/quick_test
```

### 🔍 验证结果
训练完成后，检查以下文件：
- `output/quick_test/point_cloud/iteration_7000/point_cloud.ply` - 3D点云模型
- `output/quick_test/test/` - 渲染的测试图像
- `output/quick_test/cfg_args` - 训练配置

**🎉 恭喜！你已经成功运行了第一个GeoTrack-GS模型！**

---

## 主要功能 (Features)

### 🎯 核心技术特性
*   **高质量实时渲染**: 基于 3D Gaussian Splatting，实现照片级的实时渲染效果
*   **GT-DCA外观增强**: 集成GT-DCA（Geometry-guided Track-based Deformable Cross-Attention）模块，提供增强的外观建模能力
*   **几何约束系统**: 集成多尺度几何约束，提升重建精度和一致性
*   **PyTorch 核心**: 完全使用 PyTorch 构建，易于理解、修改和扩展

### 🚀 GT-DCA 增强外观建模
*   **两阶段处理流程**: 几何引导 + 可变形采样的完整外观建模管道
*   **几何引导模块**: 利用2D轨迹点通过交叉注意力机制注入几何上下文
*   **可变形采样模块**: 预测采样偏移量和权重，从2D特征图进行自适应采样
*   **轨迹质量评估**: 智能评估和管理特征轨迹质量，确保建模稳定性
*   **性能优化**: 支持特征缓存、混合精度训练和内存优化策略

### 🔧 几何约束系统
*   **自适应权重调整**: 动态调整约束权重，优化训练过程
*   **轨迹管理**: 智能相机轨迹管理，提升多视角一致性
*   **重投影验证**: 实时重投影误差验证，确保几何准确性
*   **多尺度约束**: 在多个分辨率尺度上保持几何约束的一致性

### 🎯 几何先验各向异性正则化 (NEW!)
*   **局部几何感知**: 通过PCA分析K近邻高斯基元，提取局部几何结构
*   **各向异性约束**: 自适应地正则化每个高斯基元的形状，使其与局部几何对齐
*   **三重约束机制**: 主轴对齐 + 尺度比例约束 + 过度各向异性惩罚
*   **视图稀疏优化**: 显著减少视图稀疏场景下的边缘模糊和细节损失

### 🛠️ 工程特性
*   **端到端工作流**: 支持从 COLMAP 数据集直接进行训练、渲染和评估
*   **置信度评估**: 集成置信度信息，提升渲染的稳定性和准确性
*   **模块化设计**: 清晰的接口设计，支持独立使用和扩展
*   **错误处理**: 完善的降级机制，确保系统稳定性


## 安装 (Installation)

### � COLMAP 源码编译安装 (Linux服务器)

在Linux服务器部署时，需要手动源码编译安装COLMAP，不能使用apt-get等包管理器（兼容性问题）。默认启用CUDA支持。

#### 🔧 系统依赖安装

```bash
# 更新系统包管理器
sudo apt-get update

# 安装基础编译工具和依赖
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# 如果遇到GCC版本过低的问题，更新GCC
sudo apt-get install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

#### 🏗️ COLMAP 源码编译

```bash
# 1. 克隆COLMAP源码
git clone https://github.com/colmap/colmap.git
cd colmap

# 2. 创建构建目录
mkdir build
cd build

# 3. 配置CMake构建
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON (开启CUDA加速)

# 4. 编译COLMAP (使用多线程加速)
make -j$(nproc)

# 5. 安装COLMAP到系统
sudo make install

# 6. 验证安装
colmap -h
```

#### 🐛 常见编译问题及解决方案

**问题1: GCC版本过低**
```bash
# 错误信息: error: 'xxx' was not declared in this scope
# 解决方案: 更新GCC到9.0+版本
sudo apt-get install -y gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
```

**问题2: Ceres依赖缺失**
```bash
# 错误信息: Could not find Ceres
# 解决方案: 手动编译安装Ceres
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

**问题3: Qt5依赖问题**
```bash
# 错误信息: Qt5 not found
# 解决方案: 安装完整的Qt5开发包
sudo apt-get install -y \
    qtbase5-dev \
    qttools5-dev \
    qttools5-dev-tools \
    libqt5opengl5-dev \
    libqt5svg5-dev

**自定义安装路径:**
```bash
# 安装到自定义目录（避免需要sudo权限）
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/colmap
make -j$(nproc)
make install

# 添加到PATH环境变量
echo 'export PATH=$HOME/colmap/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### ✅ 安装验证

```bash
# 验证COLMAP安装成功
colmap -h

# 检查COLMAP版本
colmap --version

# 测试COLMAP功能
colmap feature_extractor --help
colmap exhaustive_matcher --help
colmap mapper --help
```


### 🚀 快速安装

#### 1. 克隆仓库

```bash
# 注意：--recursive 参数是必需的，用于克隆所有子模块
git clone --recursive https://github.com/CPy255/GeoTrack-GS.git
cd GeoTrack-GS
```

#### 2. 创建并激活 Conda 环境

```bash
# 创建环境
conda env create -f environment.yml
conda activate geotrack

# 或者手动创建环境
conda create -n geotrack python=3.9 -y
conda activate geotrack
```

#### 3. 安装 PyTorch 和依赖

```bash
# 自动检测并安装合适的 PyTorch 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 验证 GPU 支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

#### 4. 构建扩展模块

```bash
# 构建差分高斯光栅化模块（支持置信度）
cd submodules/diff-gaussian-rasterization-confidence
pip install .
cd ../..

# 构建简单 KNN 模块
cd submodules/simple-knn
pip install .
cd ../..
```

## 使用方法 (Usage)

### 1. 准备数据

#### 数据集要求
你需要一个由 COLMAP 处理过的数据集。目录结构通常应包含一个 `images` 文件夹和一个带有 COLMAP 重建结果的 `sparse` 文件夹。

#### 数据预处理 - COLMAP 处理

如果你有自己的图像数据，需要使用 COLMAP 进行预处理。本项目支持两种主要的数据集格式：

##### LLFF 数据集处理
对于前向面向场景（如街景、建筑物正面等），使用 LLFF 格式：

```bash
# 安装 COLMAP (如果尚未安装)
# Ubuntu/Debian:
sudo apt-get install colmap

# Windows: 下载并安装 COLMAP from https://colmap.github.io/

# 使用 tools/colmap_llff.py 脚本处理 LLFF 数据集
python tools/colmap_llff.py -s /path/to/your/images

# 指定输出路径
python tools/colmap_llff.py -s /path/to/your/images -o /path/to/output

# 高质量 LLFF 处理
python tools/colmap_llff.py -s /path/to/your/images -o /path/to/output --quality high --feature_type sift
```

##### 360度数据集处理
对于 360 度环绕场景（如物体中心拍摄），使用 360 格式：

```bash
# 处理 360 度数据集
python tools/colmap_360.py -s /path/to/your/images

# 指定输出路径
python tools/colmap_360.py -s /path/to/your/images -o /path/to/output

# 对于高质量 360 度重建
python tools/colmap_360.py -s /path/to/your/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --matcher_type exhaustive
```

##### 手动 COLMAP 处理
如果需要更精细的控制，可以手动运行 COLMAP：

```bash
# 1. 特征提取
colmap feature_extractor \
    --database_path /path/to/database.db \
    --image_path /path/to/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1

# 2. 特征匹配
colmap exhaustive_matcher \
    --database_path /path/to/database.db \
    --SiftMatching.use_gpu 1

# 3. 稀疏重建
colmap mapper \
    --database_path /path/to/database.db \
    --image_path /path/to/images \
    --output_path /path/to/sparse

# 4. 图像去畸变（可选，用于密集重建）
colmap image_undistorter \
    --image_path /path/to/images \
    --input_path /path/to/sparse/0 \
    --output_path /path/to/dense \
    --output_type COLMAP
```

##### 数据集质量优化建议

**拍摄建议：**
- **LLFF 场景**: 保持相机朝向一致，适度的视差变化
- **360 场景**: 围绕物体均匀拍摄，保持距离一致
- **重叠度**: 相邻图像至少 60-80% 重叠
- **光照**: 保持一致的光照条件
- **分辨率**: 推荐 1080p 或更高分辨率

**COLMAP 参数优化：**

```bash
# LLFF 高质量处理
python tools/colmap_llff.py -s /path/to/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --num_threads 8

# 360 度高质量处理
python tools/colmap_360.py -s /path/to/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --matcher_type exhaustive

# 针对困难场景的参数
python tools/colmap_llff.py -s /path/to/images \
    -o /path/to/output \
    --quality extreme \
    --feature_type sift \
    --matcher_type exhaustive \
    --ba_refine_focal_length \
    --ba_refine_principal_point
```

##### 处理示例

**完整的数据处理流程：**

```bash
# 1. LLFF 数据集处理示例
python tools/colmap_llff.py -s /path/to/llff/images -o /path/to/llff/output

# 2. 360 度数据集处理示例
python tools/colmap_360.py -s /path/to/360/images -o /path/to/360/output

# 3. 批量处理多个数据集
for dataset in dataset1 dataset2 dataset3; do
    python tools/colmap_llff.py -s /path/to/$dataset/images -o /path/to/$dataset/processed
done

# 4. 处理后直接训练
python tools/colmap_llff.py -s /path/to/images -o /path/to/processed
python train.py -s /path/to/processed -m output/model
```

### 2. 训练

#### 基础训练
使用 `train.py` 脚本来训练一个新模型：

```bash
# 基础训练示例
python train.py -s data/tandt/train -m output/tandt

# 高质量训练（更多迭代）
python train.py -s data/tandt/train -m output/tandt_hq --iterations 30000

# 使用几何约束的训练
python train.py -s data/tandt/train -m output/tandt_geo --enable_geometric_constraints --constraint_weight 0.1
```

#### GT-DCA增强外观建模训练
启用GT-DCA模块进行增强的外观建模：

```bash
# 启用GT-DCA增强外观建模
python train.py -s data/tandt/train -m output/tandt_gtdca \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8

# GT-DCA高质量训练
python train.py -s data/tandt/train -m output/tandt_gtdca_hq \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_attention_heads 8 \
    --gt_dca_enable_caching

# 结合几何约束和GT-DCA的完整训练
python train.py -s data/tandt/train -m output/tandt_full \
    --enable_geometric_constraints \
    --use_gt_dca \
    --multiscale_constraints \
    --adaptive_weighting \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --constraint_weight 0.1 \
    --iterations 25000
```

#### 几何约束训练
启用几何约束系统进行更精确的重建：

```bash
# 启用多尺度几何约束
python train.py -s data/tandt/train -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --constraint_weight 0.15 \
    --adaptive_weighting

# 启用轨迹管理和重投影验证
python train.py -s data/tandt/train -m output/tandt_trajectory \
    --enable_geometric_constraints \
    --trajectory_management \
    --reprojection_validation \
    --constraint_weight 0.2

# 完整几何约束训练（推荐用于高质量重建）
python train.py -s data/tandt/train -m output/tandt_full_geo \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --adaptive_weighting \
    --trajectory_management \
    --reprojection_validation \
    --constraint_weight 0.1 \
    --iterations 25000
```

#### 几何正则化训练 (NEW!)
启用基于几何先验的各向异性正则化：

```bash
# 基础几何正则化训练
python train.py -s data/tandt/train -m output/tandt_geometry_reg \
    --geometry_reg_enabled \
    --geometry_reg_weight 0.01 \
    --geometry_reg_k_neighbors 16

# 高质量几何正则化训练  
python train.py -s data/tandt/train -m output/tandt_geometry_reg_hq \
    --geometry_reg_enabled \
    --geometry_reg_weight 0.02 \
    --geometry_reg_k_neighbors 24 \
    --geometry_reg_enable_threshold 3000 \
    --iterations 30000

# 结合所有功能的终极训练配置
python train.py -s data/tandt/train -m output/tandt_ultimate \
    --enable_geometric_constraints \
    --use_gt_dca \
    --geometry_reg_enabled \
    --multiscale_constraints \
    --adaptive_weighting \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --geometry_reg_weight 0.01 \
    --geometry_reg_k_neighbors 16 \
    --constraint_weight 0.1 \
    --iterations 30000
```

#### 训练参数说明

**基础参数:**
*   `-s, --source_path`: 输入数据集所在的目录路径
*   `-m, --model_path`: 用于保存训练好的模型和检查点的目录路径
*   `--iterations`: 训练迭代次数 (默认: 30000)

**GT-DCA参数:**
*   `--use_gt_dca`: 启用GT-DCA增强外观建模
*   `--gt_dca_feature_dim`: GT-DCA特征维度 (默认: 256)
*   `--gt_dca_num_sample_points`: 可变形采样点数量 (默认: 8)
*   `--gt_dca_hidden_dim`: GT-DCA隐藏层维度 (默认: 128)
*   `--gt_dca_attention_heads`: 交叉注意力头数 (默认: 8)
*   `--gt_dca_confidence_threshold`: 轨迹点置信度阈值 (默认: 0.5)
*   `--gt_dca_min_track_points`: 最小轨迹点数量 (默认: 4)
*   `--gt_dca_enable_caching`: 启用GT-DCA特征缓存
*   `--gt_dca_dropout_rate`: GT-DCA模块的Dropout率 (默认: 0.1)

**几何约束参数:**
*   `--enable_geometric_constraints`: 启用几何约束系统
*   `--multiscale_constraints`: 启用多尺度约束
*   `--adaptive_weighting`: 启用自适应权重调整
*   `--trajectory_management`: 启用轨迹管理
*   `--reprojection_validation`: 启用重投影验证
*   `--constraint_weight`: 几何约束权重 (默认: 0.1)

**几何正则化参数 (NEW!):**
*   `--geometry_reg_enabled`: 启用几何先验正则化
*   `--geometry_reg_weight`: 几何正则化权重 (默认: 0.01)
*   `--geometry_reg_k_neighbors`: PCA分析的K近邻数量 (默认: 16)
*   `--geometry_reg_enable_threshold`: 开始正则化的迭代阈值 (默认: 5000)
*   `--geometry_reg_min_eigenvalue_ratio`: 最小特征值比率 (默认: 0.1)

**混合精度训练参数:**
*   `--mixed_precision`: 启用全局混合精度训练（自动混合精度AMP）
*   `--amp_dtype`: 混合精度数据类型，可选fp16或bf16 (默认: fp16)
*   说明: 控制整个训练循环的混合精度，包括：
    - 渲染过程的前向传播
    - 损失计算（L1、SSIM、几何约束等）
    - 反向传播和梯度计算
    - 优化器参数更新
*   效果: 显著降低GPU内存使用（约30-50%），同时加速训练过程

要查看所有可用的训练选项，请运行：
```bash
python train.py --help
```

### 3. 渲染

#### 基础渲染
当您拥有一个训练好的模型后，就可以从新的摄像机视角渲染图像：

```bash
# 基础渲染
python render.py -m output/tandt

# 高质量渲染
python render.py -m output/tandt --render_quality high

# 渲染测试集
python render.py -m output/tandt --skip_train --skip_test
```

#### GT-DCA增强渲染
使用GT-DCA模块进行增强的外观渲染：

```bash
# 启用GT-DCA增强渲染
python render.py -m output/tandt_gtdca --use_gt_dca

# GT-DCA高质量渲染
python render.py -m output/tandt_gtdca_hq \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_enable_caching

# 结合几何约束和GT-DCA的完整渲染
python render.py -m output/tandt_full \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8
```

#### 几何约束渲染
使用几何约束进行更稳定的渲染：

```bash
# 启用几何约束渲染
python render.py -m output/tandt_geo --enable_geometric_constraints

# 启用重投影验证的渲染
python render.py -m output/tandt_geo \
    --enable_geometric_constraints \
    --reprojection_validation \
    --validation_threshold 2.0

# 多尺度约束渲染
python render.py -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints
```

#### 渲染参数说明

**基础参数:**
*   `-m, --model_path`: 训练好的模型路径
*   `--skip_train`: 跳过训练集渲染
*   `--skip_test`: 跳过测试集渲染

**GT-DCA参数:**
*   `--use_gt_dca`: 启用GT-DCA增强渲染
*   `--gt_dca_feature_dim`: GT-DCA特征维度 (默认: 256)
*   `--gt_dca_num_sample_points`: 可变形采样点数量 (默认: 8)
*   `--gt_dca_hidden_dim`: GT-DCA隐藏层维度 (默认: 128)
*   `--gt_dca_attention_heads`: 交叉注意力头数 (默认: 8)
*   `--gt_dca_enable_caching`: 启用GT-DCA特征缓存

**几何约束参数:**
*   `--enable_geometric_constraints`: 启用几何约束
*   `--reprojection_validation`: 启用重投影验证
*   `--multiscale_constraints`: 启用多尺度约束
*   `--validation_threshold`: 验证阈值 (默认: 1.0)

**混合精度渲染参数:**
*   `--mixed_precision`: 启用渲染过程的混合精度计算
*   说明: 控制渲染过程中的混合精度，包括高斯基元的前向传播和颜色计算
*   效果: 降低GPU内存使用，加速渲染速度

### 4. 评估

#### 基础评估
要评估一个训练好的模型，请使用 `full_eval.py` 脚本：

```bash
# 基础评估
python full_eval.py -m output/tandt

# 详细评估报告
python full_eval.py -m output/tandt --detailed_report
```

#### 几何约束评估
评估几何约束模型的性能：

```bash
# 几何约束评估
python full_eval.py -m output/tandt_geo \
    --enable_geometric_constraints \
    --geometric_metrics

# 多尺度约束评估
python full_eval.py -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --geometric_metrics \
    --detailed_report

# 完整评估（包含所有指标）
python full_eval.py -m output/tandt_full_geo \
    --enable_geometric_constraints \
    --geometric_metrics \
    --reprojection_metrics \
    --trajectory_metrics \
    --detailed_report
```

#### GT-DCA评估
评估GT-DCA增强外观建模的效果：

```bash
# GT-DCA模型评估
python full_eval.py -m output/tandt_gtdca

# GT-DCA高质量模型评估
python full_eval.py -m output/tandt_gtdca_hq --detailed_report

# 结合几何约束和GT-DCA的完整评估
python full_eval.py -m output/tandt_full \
    --detailed_report
```

#### 评估参数说明
*   `-m, --model_path`: 要评估的模型路径
*   `--enable_geometric_constraints`: 启用几何约束评估
*   `--geometric_metrics`: 计算几何相关指标
*   `--reprojection_metrics`: 计算重投影误差指标
*   `--trajectory_metrics`: 计算轨迹一致性指标
*   `--detailed_report`: 生成详细评估报告

### 5. 配置文件

项目支持通过配置文件自定义几何约束参数：

```bash
# 使用自定义配置文件训练
python train.py -s data/tandt/train -m output/tandt_custom \
    --config config/constraints.json

# 查看默认配置
python -c "from geometric_constraints.config import load_config; print(load_config())"
```

### 6. GT-DCA 详细说明

#### 🔬 GT-DCA 模块架构

GT-DCA（Geometry-guided Track-based Deformable Cross-Attention）是本项目的核心创新功能，采用模块化设计，包含以下核心组件：

**核心模块结构:**
```
gt_dca/
├── core/                    # 核心接口和数据结构
│   ├── interfaces.py        # 抽象接口定义
│   └── data_structures.py   # 数据结构定义
├── modules/                 # 主要功能模块
│   ├── gt_dca_module.py     # 主模块集成
│   ├── base_appearance_generator.py      # 基础外观特征生成
│   ├── geometry_guided_module.py         # 几何引导模块
│   └── deformable_sampling_module.py     # 可变形采样模块
├── integration/             # 系统集成
│   ├── gaussian_model_extension.py       # 高斯模型扩展
│   └── fallback_handler.py              # 降级处理
└── utils/                   # 工具函数
    ├── tensor_utils.py      # 张量操作工具
    ├── validation.py        # 输入验证
    └── error_handling.py    # 错误处理
```

#### 🚀 两阶段处理流程

**阶段1: 几何引导 (Geometry Guidance)**
1. **基础特征生成**: 从3D高斯基元生成可学习的基础外观特征作为查询向量
2. **几何上下文提取**: 从2D轨迹点提取几何上下文信息
3. **交叉注意力处理**: 使用多头交叉注意力机制将几何上下文注入查询向量

**阶段2: 可变形采样 (Deformable Sampling)**
1. **偏移预测**: 基于几何引导特征预测采样偏移量
2. **权重计算**: 计算每个采样点的注意力权重
3. **特征采样**: 从2D特征图进行可变形采样
4. **特征聚合**: 加权聚合采样特征生成最终的增强外观特征

#### ⚙️ 配置参数详解

**核心配置参数:**

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `feature_dim` | 256 | GT-DCA特征维度 | 64-512 |
| `hidden_dim` | 128 | MLP隐藏层维度 | 32-256 |
| `num_sample_points` | 8 | 可变形采样点数量 | 2-16 |
| `attention_heads` | 8 | 交叉注意力头数 | 2-16 |
| `confidence_threshold` | 0.5 | 轨迹点置信度阈值 | 0.3-0.8 |
| `min_track_points` | 4 | 最小轨迹点数量 | 3-10 |
| `dropout_rate` | 0.1 | Dropout率 | 0.0-0.3 |
| `enable_caching` | False | 启用特征缓存 | True/False |

#### 🎯 配置建议

**基础配置（适用于大多数场景）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --gt_dca_attention_heads 8 \
    --gt_dca_enable_caching
```

**高质量配置（适用于复杂场景）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_attention_heads 16 \
    --gt_dca_enable_caching \
    --gt_dca_dropout_rate 0.05 \
    --gt_dca_confidence_threshold 0.6
```

**内存优化配置（适用于GPU内存有限的情况）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 128 \
    --gt_dca_num_sample_points 4 \
    --gt_dca_attention_heads 4 \
    --gt_dca_dropout_rate 0.2 \
    --mixed_precision \
    --amp_dtype fp16
```

**Tesla T4 优化配置（16GB显存）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 64 \
    --gt_dca_hidden_dim 32 \
    --gt_dca_num_sample_points 2 \
    --gt_dca_attention_heads 2 \
    --gt_dca_confidence_threshold 0.8 \
    --gt_dca_enable_caching \
    --mixed_precision
```

#### 🔧 性能优化策略

**1. 内存优化**
- **特征维度**: 根据GPU显存调整 `feature_dim` (64-512)
- **采样点数**: 减少 `num_sample_points` 可显著降低内存使用
- **混合精度**: 启用 `--mixed_precision` 开启全局AMP训练，减少显存占用并加速训练
- **缓存策略**: 合理使用 `--gt_dca_enable_caching`

**2. 计算优化**
- **注意力头数**: 平衡质量和速度，推荐 4-8 个头
- **Dropout**: 训练时使用 0.1，推理时设为 0.0
- **批处理**: 通过轨迹点过滤减少计算量

**3. 质量优化**
- **置信度阈值**: 提高阈值可过滤低质量轨迹点
- **最小轨迹点**: 确保足够的几何约束信息
- **特征维度**: 更高维度通常带来更好质量


#### 🐛 故障排除

**常见问题及解决方案:**

1. **轨迹点不足错误**
```bash
❌ 有效轨迹点数量(2)少于最小要求(4)！
```
**解决方案:**
- 降低 `--gt_dca_confidence_threshold`
- 减少 `--gt_dca_min_track_points`
- 检查轨迹文件质量

2. **内存不足错误**
```bash
RuntimeError: CUDA out of memory
```
**解决方案:**
- 减少 `--gt_dca_feature_dim`
- 降低 `--gt_dca_num_sample_points`
- 启用 `--mixed_precision` 进行全局混合精度训练

3. **性能问题**
```bash
GT-DCA处理速度过慢
```
**解决方案:**
- 启用 `--gt_dca_enable_caching`
- 减少 `--gt_dca_attention_heads`
- 优化轨迹点过滤阈值

### 7. 几何先验各向异性正则化详解 (NEW!)

#### 🎯 技术原理

几何先验各向异性正则化是本项目的最新创新功能，旨在解决标准3DGS在视图稀疏情况下的形态不匹配问题。

**核心思想：**
- 在标准3DGS中，高斯基元的形状仅受渲染颜色的隐式监督
- 视图稀疏时，可能出现"胖"椭球表示薄平面的情况，导致边缘模糊
- 通过引入局部几何结构作为先验知识，显式约束高斯形状

#### 🔬 算法流程

**两阶段处理：**

1. **局部几何感知 (Local Geometry Perception)**
   - 对每个高斯基元，寻找其K个最近邻高斯基元
   - 对邻居位置进行主成分分析(PCA)，提取局部主方向
   - 获得局部几何结构的特征值和特征向量

2. **各向异性约束 (Anisotropic Constraint)**
   - **主轴对齐约束**: 使高斯主轴与局部几何主方向对齐
   - **尺度比例约束**: 调整高斯尺度比例匹配局部几何特征值比例
   - **过度各向异性惩罚**: 防止高斯过度拉伸，保持稳定性

#### ⚙️ 参数配置详解

**核心参数说明:**

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `geometry_reg_weight` | 0.01 | 正则化权重 | 0.005-0.05 |
| `geometry_reg_k_neighbors` | 16 | K近邻数量 | 8-32 |
| `geometry_reg_enable_threshold` | 5000 | 启用迭代阈值 | 3000-7000 |
| `geometry_reg_min_eigenvalue_ratio` | 0.1 | 最小特征值比率 | 0.05-0.2 |

#### 🎯 使用建议

**基础使用场景：**
```bash
# 视图稀疏的室内场景
python train.py -s data/indoor_scene -m output/indoor_reg \
    --geometry_reg_enabled \
    --geometry_reg_weight 0.015 \
    --geometry_reg_k_neighbors 20
```

**高质量重建场景：**
```bash
# 建筑物外墙等平面丰富的场景
python train.py -s data/building -m output/building_reg \
    --geometry_reg_enabled \
    --geometry_reg_weight 0.025 \
    --geometry_reg_k_neighbors 24 \
    --geometry_reg_enable_threshold 3000
```

**与其他功能结合：**
```bash
# 结合GT-DCA和几何约束的完整配置
python train.py -s data/complex_scene -m output/complete \
    --enable_geometric_constraints \
    --use_gt_dca \
    --geometry_reg_enabled \
    --gt_dca_feature_dim 256 \
    --geometry_reg_weight 0.01 \
    --geometry_reg_k_neighbors 16 \
    --constraint_weight 0.1
```


#### 🔧 调优指南

**内存优化：**
- 减少`k_neighbors`数量可显著降低内存使用
- 适合GPU内存有限的场景

**质量优化：**
- 增加`geometry_reg_weight`可增强正则化效果
- 降低`enable_threshold`可更早开始正则化

**稳定性优化：**
- 调整`min_eigenvalue_ratio`防止数值不稳定
- 渐进式增加正则化权重

#### 🐛 常见问题

**Q: 几何正则化损失为0？**
A: 检查`enable_threshold`设置，确保训练已达到启用阈值

**Q: 训练速度明显变慢？**
A: 减少`k_neighbors`或启用simple_knn加速

**Q: 正则化效果不明显？**  
A: 适当增加`geometry_reg_weight`，或降低`enable_threshold`

### 8. 可视化分析

项目提供了完整的可视化分析工具，详细使用方法请参考 [Visualization_Usage.md](Visualization_Usage.md)。

主要功能包括：
- 轨迹特征可视化分析
- 几何正则化效果展示  
- GT-DCA增强外观建模可视化
- 高性能轨迹查询系统

```bash
# 快速开始轨迹可视化
python run_trajectory_visualization.py --tracks_h5 /path/to/tracks.h5

# 详细使用方法请查看可视化使用文档
cat Visualization_Usage.md
```

---


## ❓ 常见问题 (FAQ)

### 🔧 安装相关问题

**Q: COLMAP编译失败，提示GCC版本过低？**
```bash
# 错误信息: error: 'xxx' was not declared in this scope
# 解决方案: 更新GCC到9.0+版本
sudo apt-get install -y gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
```

**Q: 找不到Ceres依赖？**
```bash
# 错误信息: Could not find Ceres
# 解决方案: 手动编译安装Ceres
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

**Q: PyTorch CUDA版本不匹配？**
```bash
# 如果 CUDA 版本不匹配，重新安装对应版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Q: 扩展模块编译失败？**
```bash
# 清理缓存后重新编译
pip cache purge
python -m pip install --upgrade pip setuptools wheel

# 如果内存不足，使用单线程编译
export MAX_JOBS=1
cd submodules/diff-gaussian-rasterization-confidence
pip install .
```

### 🚀 训练相关问题

**Q: GT-DCA训练时出现"轨迹点不足"错误？**
```bash
❌ 有效轨迹点数量(2)少于最小要求(4)！
```
**解决方案:**
- 降低置信度阈值: `--gt_dca_confidence_threshold 0.3`
- 减少最小轨迹点: `--gt_dca_min_track_points 2`
- 检查数据集质量和COLMAP重建结果

**Q: 训练时GPU内存不足？**
```bash
RuntimeError: CUDA out of memory
```
**解决方案:**
- 减少GT-DCA特征维度: `--gt_dca_feature_dim 128`
- 降低采样点数量: `--gt_dca_num_sample_points 4`
- 启用全局混合精度训练: `--mixed_precision`
- 使用内存优化配置（见文档Tesla T4配置）

**Q: 几何正则化损失始终为0？**
**解决方案:**
- 检查启用阈值设置: `--geometry_reg_enable_threshold 3000`
- 确保训练迭代数超过阈值
- 验证K近邻设置: `--geometry_reg_k_neighbors 16`

**Q: 训练速度过慢？**
**解决方案:**
- 启用GT-DCA缓存: `--gt_dca_enable_caching`
- 减少注意力头数: `--gt_dca_attention_heads 4`
- 降低几何正则化K近邻数量
- 使用更少的训练迭代进行测试

### 📊 数据处理问题

**Q: COLMAP重建失败？**
```bash
# 降低质量要求
python tools/colmap_llff.py -s /path/to/images --quality medium --feature_type orb

# 如果图像过多导致内存不足
python tools/colmap_360.py -s /path/to/images --max_num_images 200 --quality medium

# 使用顺序匹配替代穷举匹配
python tools/colmap_llff.py -s /path/to/images --matcher_type sequential --overlap 10
```

**Q: 图像质量不佳，如何优化？**
**拍摄建议:**
- 保持60-80%的图像重叠度
- 使用一致的光照条件
- 推荐1080p或更高分辨率
- LLFF场景保持相机朝向一致
- 360场景围绕物体均匀拍摄

### 🎯 性能优化问题

**Q: 如何针对不同GPU优化配置？**

**RTX 4090 (24GB) - 高质量配置:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --enable_geometric_constraints \
    --geometry_reg_enabled \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_attention_heads 16 \
    --geometry_reg_k_neighbors 24 \
    --iterations 30000
```

**RTX 3080 (10GB) - 平衡配置:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --enable_geometric_constraints \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --gt_dca_attention_heads 8 \
    --gt_dca_enable_caching \
    --iterations 25000
```

**GTX 1080 Ti (11GB) - 内存优化配置:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 128 \
    --gt_dca_num_sample_points 4 \
    --gt_dca_attention_heads 4 \
    --mixed_precision \
    --iterations 20000
```

### 🔍 调试和监控


**Q: 如何检查模型质量？**
```bash
# 快速评估
python full_eval.py -m output/your_model

# 详细评估报告
python full_eval.py -m output/your_model --detailed_report

# 检查渲染结果
python render.py -m output/your_model
# 查看 output/your_model/test/ 目录下的渲染图像
```

**Q: 训练中断后如何恢复？**
```bash
# GeoTrack-GS支持自动从检查点恢复训练
python train.py -s /path/to/dataset -m output/existing_model

# 系统会自动检测并从最新检查点继续训练
# 检查点保存在: output/model/chkpnt*.pth
```


## 如何贡献 (Contributing)

欢迎任何形式的贡献！如果你发现了 bug 或有功能建议，请随时通过 [GitHub Issues](https://github.com/CPy255/GeoTrack-GS/issues) 提出。

## 致谢 (Acknowledgements)

本项目基于以下出色的工作：
*   [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
*   [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

感谢原作者的杰出贡献。

## 许可证 (License)

本项目根据 [LICENSE.md](LICENSE.md) 文件中的条款进行许可。
