# E-Manifold-GS 项目中文 Framework

## 1. 项目一句话定义

这是一个围绕 **3D Gaussian Splatting 几何约束化** 展开的研究型工程项目。它的核心目标不是单纯把 3DGS 跑起来，而是把传统 3DGS 里“自由漂浮的 3D 高斯”重新解释成一种更接近 **曲面测度 / 流形采样** 的表示，并进一步把这种几何解释接到：

- 训练期的几何正则；
- 训练后的离线流形投影；
- 网格/patch 资产化输出；
- 几何与渲染的对照评测。

用更直接的话说，这个项目在做的是：

> 先让 3DGS 的高斯不再只是渲染粒子，而是尽量变成“有切平面、有法向、有局部面积、有邻接结构”的几何原语；再基于这些几何原语抽出更稳定的表面骨架与资产输出。

## 2. 项目定位

这个仓库不是纯论文草稿，也不是纯训练脚本，而是四层并行推进的混合型项目：

1. **理论层**
   定义“什么时候一组高斯可以被认为接近某个二维流形”。

2. **算法层**
   把高斯协方差解释成局部几何，再设计 manifold-conservative 的损失与投影方法。

3. **工程层**
   在官方 3DGS 代码之上，以尽量小的侵入方式接入训练 hook、离线分析、patch mesh 提取和评测脚本。

4. **实验层**
   用解析几何场景、manifest 编排、GT 几何指标和渲染指标来验证方法是否真的比 vanilla 3DGS 更稳定。

## 3. 核心研究问题

项目的中心问题可以拆成三步：

1. **解释问题**
   给定一个 3DGS 高斯 `(\mu, \Sigma, \alpha, c)`，是否可以从协方差 `\Sigma` 中恢复它更像曲面片、曲线段还是体元？

2. **约束问题**
   如果一个高斯被解释为曲面片，那么相邻高斯之间是否应该满足局部二维流形的一致性，例如法向连续、局部秩接近 2、面积密度平滑、曲率与尺度相容？

3. **输出问题**
   如果这些约束成立，是否能够从高斯集合中稳定地产出：
   - patch graph；
   - patch mesh；
   - 投影后的 oriented points；
   - 后续更适合编辑/仿真/资产化的几何骨架？

## 3.5 研究背景、相关工作与我们的定位

如果把 Gaussian Splatting 的几何方向放进一个更清晰的研究背景里，大致可以分成下面几类。

### A. 原始 3DGS 路线：先解决高质量新视角渲染

代表：`3D Gaussian Splatting for Real-Time Radiance Field Rendering`

这条线的核心目标是：

- 高保真新视角合成；
- 实时渲染；
- 通过 densification 和 anisotropic Gaussian 表示提升效率与效果。

但它的天然弱点也很明显：

- 高斯更像“可优化的辐射粒子”，不是显式表面原语；
- 几何经常出现 floaters、双层、厚壳、模糊支撑；
- 直接从 checkpoint 中抽网格或抽稳定表面通常不够可靠。

所以，后续几乎所有几何向工作，都是在回答一个问题：

> 如何让 Gaussian 表示既保留 3DGS 的渲染优势，又拥有更稳定、更可解释的几何结构？

### B. 2D / Surface Primitive 路线：把单个高斯先做成“更像表面”

代表：

- `2DGS`
- `2D-SuGaR`

这类方法的典型思路是：

- 不再把 primitive 视为一般 3D 体高斯；
- 而是直接把 primitive 做成盘状 / 薄片状 / surface-aware 的局部元素；
- 从 primitive 级别强化几何一致性与可重建性。

这条路线的价值非常直接：

- 单个 primitive 的几何解释更强；
- 更容易得到法向一致、视角一致的表面；
- 对 mesh reconstruction 更友好。

但它的主要视角仍然偏向：

- **单个 primitive 应该长什么样**；
- 或 **如何更好地从这些 surface-aware primitive 中提网格**。

### C. 几何约束训练路线：给 3DGS 加 normal / depth / curvature / manifold 先验

代表：

- `PGSR`
- `GausSurf`
- `SolidGS`
- `GeoSplat`

这一类工作和我们关系最近，因为它们都在做“训练期几何规约”。

它们的共性是：

- 在 3DGS 优化过程中加入几何监督或几何正则；
- 使用 depth、normal、MVS、多视图一致性、局部 manifold、曲率等信号；
- 希望减少几何伪影，并提升 surface reconstruction 或 rendering quality。

其中可以进一步区分：

- `PGSR / GausSurf`
  更强调 MVS、法向先验或多视图几何信号来提升表面质量。

- `SolidGS`
  更强调让几何支撑更 solid、更连续、更适合 surface extraction。

- `GeoSplat`
  走得更远，开始系统性地把一阶和二阶几何量引入初始化、梯度更新和 densification，属于“几何约束 3DGS 框架化”的代表。

换句话说，如果只说“我们也用了 manifold / curvature / normal / geometry prior”，那是不够的，因为这条线上已经有不少工作。

### D. Mesh-Aligned / Mesh-Driven / Manipulation 路线：把高斯和显式网格绑定

代表：

- `SuGaR`
- `MeshGS`
- `Mani-GS`

这类工作通常把 mesh 当成更强的几何锚点，用来做：

- 网格提取与 refinement；
- mesh-aligned splat；
- 可编辑、可操控、可变形的 Gaussian 表示。

这条路线已经证明了一件事：

> 只要把 Gaussian 和显式表面结构绑定起来，资产化、编辑性和可控性都会明显增强。

但它们的着力点通常是：

- 已有 mesh 的对齐与增强；
- 从 Gaussian 到 mesh 的提取质量；
- manipulation / deformation / editing。

它们不一定把“整个 Gaussian 集合是否可被解释为一个流形诱导测度”当作中心问题。

### E. 我们项目当前最合理的定位

结合仓库当前实现和已有相关工作，`E-Manifold-GS` 最合理的定位不是：

- 第一个做几何引导的 GS；
- 第一个做 surface-aware GS；
- 第一个做 mesh extraction 的 GS；
- 第一个做 curvature / manifold regularization 的 GS。

这些说法都不稳，甚至很容易被现有工作反驳。

我们更稳妥、也更贴近仓库现状的定位是：

> 我们关注的不是“单个高斯怎么更像表面”，也不只是“如何加一些几何 prior 提升训练”，而是把**整组高斯**解释为一种**流形诱导的离散几何测度**，并围绕这个解释同时建立：
> 1. 协方差到几何的统一解释；
> 2. 邻域级流形有效性诊断；
> 3. 训练期 manifold-conservative loss；
> 4. 训练后 manifold projection；
> 5. patch mesh / 资产化输出；
> 6. 基本形式兼容性与测度守恒分析。

这意味着我们的重点是：

- **集合级 / 结构级解释**，而不是只看单个 primitive；
- **训练内 + 训练后** 的统一闭环，而不是只做其中一段；
- **几何测度、局部流形有效性、兼容性诊断**，而不是只追求更高的 PSNR 或更干净的 mesh。

## 3.6 与代表性工作的对照

下面这张对照表，更适合直接拿去做汇报时解释“我们和谁像、又和谁不一样”。

| 方法 | 主要目标 | 核心做法 | 和我们的重合点 | 和我们的关键差异 |
| --- | --- | --- | --- | --- |
| `3DGS` | 高质量实时新视角渲染 | 各向异性 3D Gaussian + densification + visibility-aware rendering | 都以 Gaussian scene 为基础 | 3DGS 不把高斯集合当作流形结构来解释 |
| `2DGS` | 让 primitive 更接近表面元素 | 使用 2D / disk-like Gaussian primitive | 都强调表面化表示 | 2DGS 更偏 primitive 设计；我们更偏集合级流形解释 |
| `SuGaR / 2D-SuGaR` | 从 Gaussian 中恢复更准确 mesh | surface-aware regularization + mesh extraction/refinement | 都关心资产化与网格质量 | 我们更强调训练期/离线期统一的流形诊断与投影闭环 |
| `PGSR / GausSurf` | 用几何先验改善表面重建 | MVS、normal prior、多视图几何引导 | 都使用 geometry guidance | 我们更强调协方差几何解释和邻域流形有效性，而不只是外部先验注入 |
| `SolidGS` | 获得更 solid 的表面几何 | 几何连续性/实体性相关正则 | 都在减少几何伪影 | 我们更明确建模 surface/curve/volume 维度类别与测度守恒 |
| `GeoSplat` | 框架化地将一阶/二阶几何量纳入 GS 训练 | 几何初始化、动态几何先验、曲率相关更新与 densification | 都不是只看 photometric loss；都强调高阶几何 | GeoSplat 更像“几何约束优化框架”；我们更强调 manifold-induced measure、训练后投影和 compatibility diagnostics |
| `MeshGS` | 让高斯和 mesh 更紧密对齐 | mesh-aligned Gaussian 绑定与渲染 | 都关心 mesh 与 splat 的桥接 | MeshGS 依赖 mesh 对齐；我们更强调从 Gaussian 自身恢复局部流形结构 |
| `Mani-GS` | 支持操控、变形和编辑 | 用三角网格驱动 Gaussian manipulation | 都重视资产化与显式结构 | Mani-GS 偏编辑/变形，不以流形测度解释和几何诊断为主 |

这里必须区分“概念对照”和“已运行 baseline”。历史 manifest 中名为
`tangent` 的方法，仅开启协方差法向到局部 support 法向的一阶监督；它是项目内部
消融，不是 GeoSplat 的复现，也不能作为 GeoSplat 的数值代理。外部方法是否已经
接入及能否进入主结果表，以 `BASELINE-STATUS.md` 为准。

## 3.7 我们当前可以主张的贡献

基于现有仓库内容，更稳妥的贡献表述建议写成下面这几条。
证据等级和论文最终口径以 `CLAIM-EVIDENCE-ZH.md` 为准；以下条目描述的是框架组成，
“已实现”不自动意味着“已稳定改善重建”或“已达到 SOTA”。

### 贡献 1：把高斯协方差解释扩展为集合级几何协议

不是只看单个高斯是否薄，而是系统性定义：

- 法向；
- 切向面积；
- surface / curve / volume 维度标签；
- 局部邻域流形有效性分数。

这构成了后续训练、投影、mesh 提取和评测的统一中间表示。

### 贡献 2：建立训练期与离线期打通的 manifold-conservative 闭环

当前仓库不是只做 loss，也不是只做后处理，而是两条路径同时存在：

- 训练期：`ManifoldLossController` 把 thinness、area、rank2、curvature、normal consistency 等约束接入 3DGS；
- 离线期：`project_points_to_manifold` + `build_patch_mesh_from_points` 把结果整理成更稳定的曲面支撑和 patch asset。

这个“在线优化 + 离线投影资产化”的组合，是仓库里非常明确的一条主线。

### 贡献 3：把“是否真像曲面”做成可计算诊断，而不是口头描述

当前实现里不只是输出 mesh 或 point cloud，还显式输出：

- graph diagnostics；
- normalized kernel varifold；
- support / predicted normal compatibility；
- symmetry / Gauss / Codazzi residual；
- 几何测度守恒检验。

这使得项目不只是在做“看起来更薄”的经验工程，而是在尝试建立一套更可证伪的几何验证协议。

### 贡献 4：强调混合维度表示，而不是强迫所有高斯都表面化

仓库中的基础叙事并不是“所有高斯都必须是 2D surface splat”，而是允许：

- surface-like primitive；
- curve-like primitive；
- volume-like primitive。

这点很重要，因为它给项目留下了处理细线结构、非刚性薄结构和真实非表面残差的空间。

## 3.8 我们不该过度主张的地方

为了让文档和后续论文口径更稳，下面这些说法建议避免：

- “我们是第一个把 GS 做成表面表示的方法”
- “我们是第一个做几何约束 GS 的方法”
- “我们首次把 manifold 引入 Gaussian Splatting”
- “我们已经显著超越 SuGaR / GeoSplat / GausSurf / SolidGS”

更合适的表述是：

- 我们提出一个 **manifold-conservative / manifold-induced measure** 的统一解释框架；
- 我们把 **集合级流形诊断、训练期几何约束、训练后投影和资产化** 接到了同一条流程中；
- 当前证据主要证明 **工程路径可行、诊断信号可计算、若干局部几何指标可优化**；
- 更强的 SOTA 结论仍需要系统 benchmark 支撑。

## 4. 总体架构图

项目可以按下面这条主线理解：

```text
输入图像 / 相机 / 可选几何先验
    ->
3DGS 初始训练或已有 checkpoint
    ->
读取 point_cloud.ply
    ->
高斯协方差几何解释
    ->
局部表面/曲线/体分类
    ->
局部邻域图与流形诊断
    ->
两条分支并行：
    A. 训练期 manifold loss hook
    B. 训练后离线 manifold projection + patch mesh
    ->
几何评测 / 渲染评测 / 实验汇总
```

如果按工程组件来画，可以理解为：

```text
third_party/gaussian-splatting
    提供原始训练与渲染能力

patches/*.diff
    把 manifold_gs 的训练 hook 接进官方 3DGS

manifold_gs/*
    提供几何解释、loss、投影、mesh、GT 指标、兼容性分析

scripts/*
    把分析、训练、评测、实验编排串成可执行入口

experiments/*
    存放解析场景、基准结果、manifest、smoke runs
```

## 5. 代码框架分层

### 5.1 外部参考层：`third_party/`

这个目录不是本项目的核心创新区，而是依赖与参考实现区：

- `third_party/gaussian-splatting`
  官方 3DGS 代码，是训练与渲染的基础底座。

- `third_party/SuGaR`
  作为 mesh extraction / refinement 方向的参考基线。

- `third_party/octree-TS`
  作为历史参考项目，用于吸收 octree / triangle splatting 相关思路。

这一层的角色是“提供 baseline 能力”，不是主要的研究逻辑承载层。

### 5.2 本项目核心算法层：`manifold_gs/`

这是整个仓库最核心的目录。它不是一个单模块，而是一组围绕“高斯几何化”的能力组件。

#### A. I/O 与基础表示

- `ply_io.py`
  负责读取和写出 3DGS 相关 PLY 数据。

- `mesh_io.py`
  负责三角网格 PLY 输出。

这部分是工程基础层，为后面的诊断、投影、mesh 导出服务。

#### B. 协方差到几何解释

- `diagnostics.py`
  项目最关键的入口模块之一。它完成：
  - 读取 3DGS PLY；
  - 解析 `xyz / opacity / scale / rotation`；
  - 从 scale 与 quaternion 恢复协方差主轴；
  - 计算特征值、特征向量、法向、切向面积；
  - 依据 `r12 / r23` 比例把高斯分类成 `surface / curve / volume / background`；
  - 导出 `summary.json`、`diagnostics.npz` 和 `surface_oriented_points.ply` 所需数据。

- `torch_geometry.py`
  这是 `diagnostics.py` 的 torch 版本辅助模块，给训练期使用。它负责：
  - quaternion 转旋转矩阵；
  - 高斯协方差特征值排序；
  - 在线构造 surface mask。

这两个模块共同定义了项目最底层的几何解释协议：**一个高斯先被解释成几何对象，后续约束与资产化才成立。**

#### C. 图结构与局部流形诊断

- `graph_diagnostics.py`
  在被判定为 surface-like 的高斯子集上构造 kNN 邻域，并计算：
  - `rank2_score`：邻域是否接近二维；
  - `normal_variation`：法向变化是否平滑；
  - `log_area_variation`：局部面积密度是否稳定；
  - `curvature_scale`：局部曲率和高斯尺度是否匹配。

这个模块的作用是回答：

> 单个高斯看起来像曲面片，不代表一群高斯能组成稳定曲面；必须检查它们的局部邻接结构是否真的“像流形”。

#### D. 训练期几何损失

- `losses.py`
  定义与 renderer 解耦的原型损失，包括：
  - `thinness_loss`
  - `area_measure_loss`
  - `curvature_scale_loss`
  - `rank2_neighborhood_loss`
  - `normal_consistency_loss`

这些 loss 本身并不直接改 3DGS 渲染器，而是作为“几何规则库”供训练 hook 调用。

- `training_hooks.py`
  这是训练期的总控模块，核心类是 `ManifoldLossController`。它负责：
  - 在 warmup 之后启动几何约束；
  - 周期性刷新邻域图；
  - 在 surface-like 高斯上构建活跃子图；
  - 按权重组合各种 manifold loss；
  - 支持 bootstrap thinness、compatibility ramp、proximal 投影等策略。

这一层是整个项目连接“离线几何逻辑”和“在线 3DGS 优化”的桥。

#### E. 训练后流形投影与资产生成

- `manifold_projection.py`
  负责把点集通过局部 PCA / MLS / 二次曲面拟合投影到更稳定的流形支撑上。输出 `ProjectedManifold`，其中包括：
  - 投影后点；
  - 法向；
  - 质量；
  - 置信度；
  - 是否接受为可信表面点；
  - 邻域与局部半径。

  这个模块的关键点是：**它估计的是支撑曲面，不依赖 3DGS 原始 scale 必须已经很理想。**

- `gaussian_projection.py`
  负责把投影结果写回高斯形式，或估计 kNN 面积质量。

- `patch_mesh.py`
  负责从投影后的 oriented points 中保守地建立 patch mesh。流程大致是：
  - 基于距离、法向、切向残差筛边；
  - 建局部图；
  - 按 chart 法向约束做区域生长；
  - 在局部二维参数域做 Delaunay；
  - 丢弃质量差或几何风险高的三角形。

这一层不是追求闭合 watertight mesh，而是生成 **保守的表面骨架**。

#### F. 理论一致性与度量模块

- `fundamental_compatibility.py`
  用离散方式检查支撑几何与预测法向场之间的一致性，涉及：
  - support normal；
  - shape operator；
  - symmetry residual；
  - Gauss residual；
  - normal curl；
  - Codazzi residual。

  这一块非常研究导向，说明项目不只停留在“看起来薄一些”，而是在尝试验证法向场是否真能来自一个合理曲面。

- `gt_metrics.py`
  几何 GT 评测，包括 Chamfer、法向指标、normalized kernel varifold distance 等。

- `geometric_measure.py`
  研究几何测度守恒，例如 split / merge / prune 之后的质量与矩是否守恒。

- `analytic_scene.py`
  生成解析几何场景、采样表面、做简单 CPU rasterize，为自一致性实验提供 GT。

### 5.3 脚本编排层：`scripts/`

这一层负责把 `manifold_gs` 中分散的能力拼成可直接执行的工作流。

重点脚本如下。

#### A. 分析与诊断

- `analyze_gaussians.py`
  输入一个 3DGS checkpoint 的 `point_cloud.ply`，输出：
  - `summary.json`
  - `diagnostics.npz`
  - `graph_diagnostics.npz`
  - `surface_oriented_points.ply`

这是最适合做第一步 sanity check 的入口。

#### B. 流形投影与资产导出

- `project_manifold.py`
  这是“训练后离线几何闭环”的总入口。它会：
  - 读入原始高斯；
  - 选取有效点；
  - 估计或继承 geometric mass；
  - 做 manifold projection；
  - 做质量重分配；
  - 输出 projected points / accepted points / projected gaussians；
  - 构建 patch mesh；
  - 如果给定 GT，再做几何指标评测。

它基本代表了本项目当前最完整的“资产化路径”。

#### C. 网格提取与评测

- `extract_patch_mesh.py`
  直接从高斯几何中提取 patch mesh。

- `extract_poisson_mesh.py`
  用 oriented points 走 Poisson 重建作为保底路径。

- `evaluate_geometry_gt.py`
  评估点与法向对 GT 曲面的贴合程度。

- `evaluate_mesh_gt.py`
  评估提取网格相对 GT 的几何表现。

- `evaluate_rendered_images.py`
  评估 held-out 渲染图像质量。

#### D. 理论/兼容性分析

- `analyze_fundamental_compatibility.py`
  评估支撑曲面与高斯法向场之间的基本形式兼容性，属于“证明你得到的是曲面而非偶然薄片”的增强分析工具。

#### E. 数据与实验生成

- `generate_analytic_scene.py`
  生成解析场景数据，用于构建可控 benchmark。

- `make_synthetic_colmap_scene.py`
  生成或组织 synthetic COLMAP 风格场景。

#### F. 实验编排与汇总

- `run_experiment_manifest.py`
  读取 manifest，把一个 benchmark 分成：
  - `prepare`
  - `train`
  - `evaluate`
  - `all`

  它是整个实验自动化的总调度器。

- `summarize_experiment_manifest.py`
  负责把多场景、多 seed、多方法的结果聚合，并做通过/失败检查。

#### G. 环境检查与补丁管理

- `preflight_3dgs.py`
  检查 CUDA / torch / 3DGS 扩展是否准备好。

- `check_or_apply_train_patch.sh`
  检查或应用训练补丁，把 `ManifoldLossController` 接入官方 3DGS。

### 5.4 补丁层：`patches/`

这一层说明本项目对官方 3DGS 的修改策略是 **尽量少改原仓库，改动外置化**。

- `gaussian_splatting_train_mcgs.diff`
  最小训练环路补丁。

- `gaussian_splatting_mcgs_headless.diff`
  更完整的补丁版本，包含 training hook 和 headless 兼容修复。

这种组织方式的价值是：

- 便于回溯修改点；
- 不把研究逻辑硬塞进第三方仓库；
- 方便之后同步官方 3DGS 更新。

### 5.5 测试层：`tests/`

目前测试覆盖的重点不是“整项目端到端训练”，而是核心几何模块的 smoke / correctness：

- patch mesh 基本可生成；
- manifold losses 可反传；
- training controller 可构图并产出 loss；
- analytic sphere 自一致；
- CPU rasterizer 可产出几何；
- split / merge / prune 保持测度守恒；
- MLS 投影可减少球面噪声；
- manifest summary 的规则判断可运行。

这说明项目当前更像“研究原型已具备模块级可信度”，而不是“完整大规模训练系统已经完全定型”。

## 6. 三条关键工作流

### 6.1 工作流一：训练后高斯诊断

这是最轻量的入口，适合先判断一个 checkpoint 有没有表面化潜力。

流程：

```text
point_cloud.ply
    ->
compute_diagnostics()
    ->
surface / curve / volume 分类
    ->
graph diagnostics
    ->
summary.json + oriented points
```

用途：

- 看当前高斯集合是否足够 thin；
- 看 surface-like 比例是否在上升；
- 看局部邻域是否接近二维；
- 给后续 patch mesh / Poisson 提取做输入准备。

### 6.2 工作流二：训练期 manifold-conservative 优化

这是“把几何先验回灌进 3DGS 优化”的路径。

流程：

```text
3DGS train loop
    ->
读取 xyz / scales / rotations / opacity
    ->
torch_geometry 计算特征值与法向
    ->
training_hooks 周期刷新活跃 surface 图
    ->
losses 计算 thin / area / curvature / rank2 / normal 等项
    ->
与 photometric loss 共同反传
```

这一条路径的目标不是替换 photometric loss，而是防止模型在稀疏视角下走向：

- 浮点；
- 双层表面；
- 体状糊开；
- 尺度与曲率不匹配；
- 没有可解释邻域结构。

### 6.3 工作流三：离线流形投影与资产输出

这是目前项目最接近最终产品形态的一条路径。

流程：

```text
checkpoint.ply
    ->
diagnostics 取有效高斯
    ->
project_points_to_manifold()
    ->
accepted / confidence / mass rebalance
    ->
projected points / projected gaussians
    ->
build_patch_mesh_from_points()
    ->
patch_mesh.ply + meta
    ->
GT geometry / varifold / mesh metrics
```

这一条路径强调的是：

- 即使训练期还不完美，也可以在后处理阶段把高斯集合整理成更干净的曲面支撑；
- 输出结果更适合拿来做分析、对照和资产化。

## 7. 这个项目真正的“主干逻辑”

如果只保留最关键的主干，可以浓缩成下面 6 个动作：

1. **从 3DGS 高斯中恢复局部几何含义**
   通过协方差特征值和特征向量得到切平面、法向、局部面积和维度类别。

2. **从单点几何判断升级到邻域结构判断**
   用 kNN 图检查这些高斯是否形成局部二维流形，而不是一堆独立薄片。

3. **把这些结构信号写成训练约束**
   在训练时约束 thinness、面积密度、法向一致性、局部秩、曲率尺度匹配。

4. **在训练后做流形投影**
   通过 MLS / 局部二次拟合把点重新压到更可信的支撑曲面上。

5. **从投影结果提取保守 mesh / patch**
   用局部图与参数化三角剖分得到可用的曲面骨架。

6. **用解析 GT 和成对 benchmark 验证收益**
   不只看 PSNR，还看几何误差、法向误差、varifold 距离和资产质量。

## 8. 当前项目成熟度判断

从仓库内容看，当前项目处于：

> **研究原型后期 / 系统化验证前期**

更具体地说：

- 理论叙事已经成型；
- 核心几何模块已经落地；
- 训练 hook 已经接入官方 3DGS 路径；
- 离线投影和 patch mesh 闭环已经能跑；
- 解析 benchmark 与汇总脚本已具备；
- 但“大规模真实场景、系统对比、稳定胜出”还不是这个仓库当前已经证明的事实。

这点和 `README.md` 里的表述是一致的：当前更像是 **验证工程路径和诊断信号**，而不是已经完成最终研究结论。

## 9. 推荐的阅读顺序

如果你要快速掌握项目，建议按这个顺序读：

1. `README.md`
   先理解研究问题、贡献目标和当前证据边界。

2. `PIPELINE.md`
   建立从输入到资产输出的总体流程图。

3. `IMPLEMENTATION.md`
   对照仓库结构和可执行命令。

4. `manifold_gs/diagnostics.py`
   看项目最基础的几何解释是怎么定义的。

5. `manifold_gs/losses.py` + `manifold_gs/training_hooks.py`
   看训练期约束是怎么挂进去的。

6. `manifold_gs/manifold_projection.py` + `manifold_gs/patch_mesh.py`
   看训练后几何闭环是怎么形成的。

7. `scripts/project_manifold.py`
   看最完整的一条离线资产化工作流。

8. `scripts/run_experiment_manifest.py`
   看整个实验系统是如何批量调度的。

## 10. 用一句话总结这个项目的 framework

这套 framework 的本质是：

> 以官方 3DGS 为渲染与优化底座，在其上增加一层“高斯协方差几何解释 + 局部流形结构约束 + 训练后曲面投影与 patch 资产化”的中间系统，从而把原本偏自由体渲染的高斯集合，逐步推进为可解释、可评测、可导出几何资产的表面表示。
