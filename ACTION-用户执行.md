# 用户执行 Action

更新日期：2026-07-03。这里只保留需要用户手动操作、耗时、占 GPU、需要联网、
会产生大量日志，或预计会显著消耗 token 的任务。短分析、代码修改、CPU 单测和
小规模离线评测由 Codex 继续完成。

## 协作规则

- 需要你执行的操作只以本文件中标记为 **待执行** 的 Action 为准。
- 每项必须写明目的、成本、命令/操作、预期产物和验收标准；信息未冻结时标记
  **不要执行**，避免浪费 GPU 和人工时间。
- 已完成或失败的实验保留记录，但明确写“不要重跑”。
- Blender、Unity/Unreal、数据集网页登录、许可证确认等 GUI/人工任务由你执行。
- 长 GPU sweep、完整外部 baseline、批量真实场景和会产生大日志的任务先写入这里，
  经你执行后我负责汇总、诊断和写论文。

## 当前状态

`plane_torus_sparse_v2_shape` 及后续 identifiability、depth robustness、structured
depth、affine calibration、mask-normalized missing-depth 实验均已完成。不要重复跑。

当前缺口是**真实外部 baseline**，不是继续把内部 `tangent` 消融当成 GeoSplat。

Asset bundle 的 CPU 导出链路已完成，无需你操作。当前 sphere seed-0 bundle 位于
`experiments/analytic_sphere_s0_manifold_full_1500_v3/hybrid_asset/`，包含 grouped
OBJ/PLY、attached/residual Gaussian PLY、collision candidate、source mapping 和
manifest。下一缺口是人工 DCC 导入验证与可量化编辑任务。

Observation evidence（P0.1）CPU 部分进度（2026-07-06，Codex 完成，无需你操作）：
在 sparse-support 与 frustum view/parallax/footprint 之上，新增了**遮挡感知 first-hit
可见性**（`compute_visibility_evidence`：以点云自身为遮挡体、按像素分箱求首次命中深度），
把原来的 `frustum_no_occlusion` 升级为 `first_hit_occlusion`，并给 patch 聚合加了
`insufficient_first_hit_visibility` 拒绝理由与 `--min-observation-first-hit-views` 门限。
已有 CPU 单测覆盖。**仍未完成**的三项——多视 photometric 一致性、restricted rendering
Fisher/Jacobian 证书、以及下游 asset benchmark——需要 GPU/图像/训练，见 Action A2--A5。
在这三项闭环前，当前 cache 仍只能叫 realizability + 几何观测支持，不得称完整
identifiability certificate。

## Action A1：Blender 导入与人工完整性检查（待执行，低算力/需 GUI）

**目的：**确认标准 OBJ/PLY 能被真实 DCC 工具读取，patch 分组、开放边界和 collision
candidate 没有导出层面的损坏。这只是工程 gate，不是论文性能实验。

**成本：**约 10--20 分钟，不需要 GPU，不消耗训练 token；需要本机 Blender GUI。

**输入目录：**

```text
/mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS/
experiments/analytic_sphere_s0_manifold_full_1500_v3/hybrid_asset/
```

**操作：**

1. 删除上次导入的旧对象，重新导入新生成的 `certified_patches.obj`；确保同目录下
   `certified_patches.mtl` 未被移动。
2. 检查 Outliner 或 material slots 中能否看到 `patch_XXXX`。新版 OBJ 同时写入
   `o/g/usemtl`，并为 28 个 patch 生成不同颜色；具体显示为多个 object 还是一个 object
   的多个 material slots 取决于 Blender OBJ importer 的 Split by Object/Group 设置。
3. 再导入 `collision_candidate.ply`，确认主体球面与 patch mesh 空间对齐。新版 collision
   candidate 会有意拒绝异常大的 `patch_0027`，因此旁边的大网格片只应出现在完整
   certified mesh 中，不应出现在 collision candidate 中。
4. 打开 wireframe，确认没有明显跨球面的长三角；开放边界允许存在。
5. 保存为 `hybrid_asset/blender_import_check.blend`，并截两张图：solid 与 wireframe，
   文件名分别为 `blender_solid.png`、`blender_wireframe.png`。

**验收：**导入无报错；两层空间对齐；无明显长跨面；能保存 `.blend`。完成后只需告诉
我“PASS”或报错文本，我会继续写自动 edit benchmark。

## Action A2：真实场景 asset benchmark（不要执行，协议尚未冻结）

该任务预计包含 DTU scan24/65/105 的 asset bundle、SuGaR/matched 3DGS extraction、
mesh simplification、编辑传播和 collision proxy 对照，会占 GPU/磁盘并产生大量日志。
当前 exporter 刚完成，指标与命令尚未冻结，**现在不要运行任何新训练或批量导出**。
我会先完成以下低成本工作：

- 自动 edit propagation metric（**已完成 2026-07-07**：`manifold_gs/edit_metrics.py` +
  `scripts/evaluate_edit_propagation.py`，把 patch 编辑经 source mapping 传播后测
  edit error / boundary leakage / residual contamination，并直接对比 certified patch
  binding 与 nearest-radius baseline；exporter 现额外写 `attached_patch_ids` 使 mapping
  自描述。合成 GT 单测已证明 certified 零泄漏、baseline 跨边界泄漏。真实场景/DCC 人工
  协议归 A5/P0.3）；
- collision candidate 的 coverage/false-surface metric（**已完成 2026-07-07**：
  `manifold_gs/collision_metrics.py` + `scripts/evaluate_collision_candidate.py`，含
  supported coverage、floater/false surface area、Hausdorff/normal error、false/missed
  collision 与 unknown-marked-free 比例，已有 CPU 单测；真实场景运行需 GT + probe 文件，
  见 A5/P0.4）；
- OBJ round-trip 与 source mapping 测试；
- manifest schema 和论文表格生成器。

完成后会把唯一可复制命令、预计时长/显存/磁盘、输出目录和 PASS/FAIL 规则补到本项，
再交给你执行。

## Action A3：多视 photometric 一致性 evidence（不要执行，等 Codex 冻结脚本）

**目的：**补齐 P0.1 证书里 first-hit 可见性之外的“被观测支持”一环——一个 patch 不仅要被
若干训练相机的 first-hit 看到，其在这些相机里的重投影颜色还必须互相一致。这才能挡住
“几何位置像曲面、但没有任何一致外观支撑”的 floater。

**CPU 分量已完成（2026-07-07，Codex）：**`manifold_gs/observation_evidence.py` 新增
`compute_photometric_evidence`（对 first-hit 可见点在各视图采样投影像素颜色、Welford 累计
跨视图 RGB 方差，<2 视图记 +inf）；cache schema 加 `photometric_std` / `photometric_view_count`
/ `photometric_mean_color`；patch 聚合与 exporter 加 `max_photometric_std` /
`min_photometric_views` 门限与 `inconsistent_photometry` / `insufficient_photometric_views`
拒绝理由；已有合成图单测。

**待你执行（真实 DTU 场景，需读全部训练原图）：**在已有 scan105 evidence 的场景上补跑
photometric 分量：

```bash
cd /root/autodl-tmp/E-Manifold-GS
python scripts/build_observation_evidence.py \
  --gaussians <scan105 matched vanilla 7k 的 point_cloud.ply> \
  --colmap-points <scan105 sparse points3D.ply> \
  --scene <scan105 COLMAP 场景根，含 sparse/0/*.txt> \
  --images <scan105 训练图像目录> \
  --out experiments/.../scan105_evidence_photometric.npz
```

输出的 `photometric_multiview_fraction` 与 `photometric_std` 分布回填 `RESULTS-LATEST.md`。
**阈值仍需冻结**：先看 std 分布再定 `--max-observation-photometric-std`，不要凭空取值；
NCC 窗口/双线性采样/遮挡容差等更强口径留待冻结后再加。IO 偏重（读全部训练图），故列此。

**最终状态：已完成（2026-07-08，Codex 直接跑，非 GPU）。** 该脚本实为 CPU/IO（迁到 3090
后本地数据齐全，14s 完成），故由 Codex 直接执行，无需你操作。scan105 `_vanilla_matched`
7k：102,783 gaussians、56 训练视图、photometric_multiview_fraction **0.9998**、std 中位
0.077 / p90 0.150 / p95 0.194。输出 `experiments/observation_evidence/scan105_photometric.npz`。
分布与阈值讨论已回填 `RESULTS-LATEST.md` §4.2。

**阈值口径已定（2026-07-08，用户选"每场景相对百分位"）：**不冻结绝对值，photometric
一致性门限改为**每场景相对百分位**。已实现 `aggregate_patch_evidence(
max_photometric_std_percentile=...)` 与 exporter 的
`--max-observation-photometric-std-percentile`（例：90 = reject 本场景 std 最差 10%
的 patch），从本场景 finite per-patch median std 现算 cap，与绝对 cap 取 min，结果写
`photometric_std_threshold` / `photometric_std_percentile` 供审计；+inf（<2 视图）patch
恒拒。有单测（含跨场景尺度自适应验证）。这样跨 scan 无需重定数值、也避免单场景过拟合。
**已端到端生效（2026-07-08）：**用 scan105 既有投影产物导出了首个真实场景 hybrid bundle
（`.../scan105_vanilla_matched/hybrid_asset/`），observation gate 首次真实作用：402 patches
中 accepted 234、insufficient_sparse_support 139、**inconsistent_photometry（相对 p90）29**。
同一 bundle 上验证 texture 颜色源：多视 `photometric_mean_color` 相对 SH-DC 把 seam 从
12.50 提到 18.36 dB（+5.86），但两者 baked seam 都 ≈ raw-color ceiling，证实剩余 seam 是
真实跨边界颜色方差、共享 atlas 修不动。完整表见 `RESULTS-LATEST.md` §4.3。

## Action A4：restricted rendering Fisher/Jacobian 证书（不要执行，协议未冻结）

**目的：**P0.1 第二版要求——估计 patch 的局部几何方向是否真的被 RGB 约束，即在受限渲染下
计算 Jacobian/Fisher information 的最小特征值。视图再多，若该方向对光度几乎无梯度，则该
几何自由度未被识别，应报低置信度而非“已识别 asset”。

**为什么现在不能跑：**需要可微渲染反传每 patch 的局部扰动，GPU 成本高且数值口径
（扰动基、正则、特征值归一化）未冻结。Codex 先在小合成场景把 Jacobian 累积与最小特征值
估计写成可测函数并给单测，再冻结真实场景命令。**现在不要在真实场景上跑任何 Fisher 扫描。**

**预期产物（冻结后）：**per-patch `min_fisher_eigenvalue` / `identified_directions`；主张口径从
“realizability-aware extraction”升级为“identified asset”的证据表。

## Action A5：下游 asset 任务 benchmark（P0.3/P0.4/P0.5，不要执行，协议未冻结）

**目的：**用可量化任务证明 hybrid asset 有用，而不只是能导出。三条线：

- **P0.3 编辑**：对选定 patch 施加刚性/非刚性形变，经 source mapping 更新绑定 Gaussian，
  测 target-region edit error、boundary leakage、residual contamination；对照 vanilla 最近邻、
  Poisson mesh、SuGaR binding 与本文 certified binding。至少一个合成有 GT + 一个真实人工协议。
- **P0.4 碰撞/未知空间**：报告 supported surface coverage、floater 面积、false/missed collision、
  简化后 Hausdorff/normal error、未知区域是否被误判为 free space。
- **P0.5 纹理 round-trip**：UV/chart atlas 或逐 patch 烘焙，报告 baking reprojection error、
  seam error、round-trip 后 PSNR/SSIM，以及 mesh-only / mesh+splat / hybrid 的质量-大小-速度 Pareto。

**CPU 度量前置已完成（2026-07-07，Codex）：**三条线的 CPU 度量与脚本都已就绪且有单测——
P0.3 `manifold_gs/edit_metrics.py` + `scripts/evaluate_edit_propagation.py`；
P0.4 `manifold_gs/collision_metrics.py` + `scripts/evaluate_collision_candidate.py`；
P0.5 `manifold_gs/texture_metrics.py` + `scripts/evaluate_texture_roundtrip.py`
（texture 脚本默认用 Gaussian SH DC 作观测色，可直接在现有 bundle 上跑，无需 A3 缓存）。

**为什么真实运行仍不能现在跑：**要训练/抽取/渲染多个真实场景并产生大日志，且指标与
baseline 选择、纹理 UV atlas / 渲染 round-trip PSNR/SSIM/LPIPS / Pareto 口径尚未冻结
（见 `PROJECT-GAPS-ZH.md` P0.3--P0.5）。Codex 下一步补表格生成器与真实场景的唯一命令、
PASS/FAIL，再交给你。**现在不要启动任何新的下游训练或批量导出。**

**2026-07-08 CPU 实跑（Codex 完成，无需你操作）：**三条度量已首次实跑在既有 sphere
seed-0 backbone 上，全部 CPU。P0.3 certified binding 零泄漏、baseline 泄漏 12.9%；
P0.4 collision 单点 coverage 26.84% / false 74.18%；P0.5 烘焙 PSNR 36.32 dB、seam 16.86 dB。
**弱点诊断（重要）：**后两个"弱点"经 CPU 诊断均为**评测口径产物、非 mechanism bug**——
collision 的 tolerance(1% bbox)比方法自身精度(中位误差 0.033)更紧，扫描到 2% bbox coverage
即 72%（bridge/alpha 过滤实测无改善，证伪切向外推假设）；texture seam 用了噪声 SH-DC 颜色，
其跨边界原始色方差 16.68 dB 已等于烘焙后 16.86 dB，共享 atlas 修不了。真正杠杆是几何训练
精度与多视 photometric 颜色（均属 GPU）。已把两处评测改成不再产生误导数值（collision 加
tolerance 扫描、texture 并列 raw ceiling）。**操作性
发现：**旧 `hybrid_asset/asset_mapping.npz` 早于 `attached_patch_ids` 字段，texture/edit
评测会拒绝旧 bundle；已重导出到 `hybrid_asset_reexport/`。**今后任何要跑 edit/texture
评测的 bundle（含真实 DTU）都必须用当前 exporter 重新导出，否则评测会报
`predates attached_patch_ids`。**

## Action 1：GeoSplat 状态（已完成）

论文为 `GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting`
（arXiv:2509.05075）。截至当前未发现官方公开代码，暂时只做论文层面对照。

## Action 2：获取并预检 2DGS（已完成）

官方仓库：`https://github.com/hbb1/2d-gaussian-splatting`。

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS/third_party
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git
```

官方仓库已下载，commit 为 `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`，两个
CUDA 子模块完整。不要直接覆盖当前 3DGS 环境；下一步需要隔离环境和正式运行命令。

## Action 3：2DGS 外部 baseline（已完成）

2DGS 协议与静态 adapter 已冻结，独立环境 `surfel_splatting` 已修复并验证。不要再
执行 `conda env create -f environment.yml`，否则会重复触发 CUDA 扩展安装问题。

当前已验证：PyTorch `2.0.0 + CUDA 11.8`、CUDA compiler `11.8`、GCC/G++ `11.4`、
`diff_surfel_rasterization` 和 `simple_knn` 均已安装。环境自检命令：

```bash
conda run -n surfel_splatting python -c \
  "import torch,diff_surfel_rasterization,simple_knn; print(torch.__version__, torch.version.cuda)"
```

plane/torus 各 3 seeds 的官方 30k 训练、rendering metrics、PLY adapter 和统一
geometry metrics 均已完成。汇总位于
`experiments/external/2dgs_plane_torus/official_30k/summary.json`。

SuGaR 不需要另写 PLY 转换器；它的 refined PLY 可直接进入现有统一 evaluator。
但官方完整流程包含 coarse optimization、mesh extraction 和 refinement，且要求 7k
3DGS checkpoint，因此未冻结命令前不要手动启动。

## Action 4：SuGaR 环境与 pilot（已完成）

SuGaR 的输入适配、固定 3/12 划分、seed、vanilla 7k checkpoint 复用及两场景 pilot
runner 已冻结。conda 主环境已经创建成功，PyTorch 2.0.1、CUDA 11.8 和 PyTorch3D
0.7.4 已就绪。原始安装卡在 pip 阶段：Open3D 0.18 wheel 大小约 399.7 MB，官方
PyPI 实测速率只有 36.7 kB/s（约 3 小时），且旧 pin 0.17 当前不可用。

不要重跑 `install.py`。Open3D 0.18、其余 Python 依赖、PyTorch3D、3DGS rasterizer
和 simple-knn 均已安装并通过导入检查。以下镜像命令已经执行完成，仅保留作记录：

```bash
conda run -n sugar python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps open3d==0.18.0
```

若清华镜像不可用，换用：

```bash
conda run -n sugar python -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --no-deps open3d==0.18.0
```

环境状态：`READY`。

两场景 pilot 已完成，以下命令仅留作记录，不要重复运行：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_sugar_pilot.py --execute
```

plane_s0 和 torus_s0 均已完成 coarse 15k + 低多边形 mesh + refinement 2k，
并生成 refined PLY、coarse mesh、统一 geometry/mesh metrics 与持久日志。

8GB 适配说明：pilot 的 SDF Monte Carlo samples 从官方 1M 降为 50k，并启用 allocator
防碎片；该结果只作 pipeline diagnostic。mesh bbox 从输入 COLMAP `points3D.ply` 范围
加 margin 得到，不使用 GT。

## 阶段策略

不要立刻扩成 SuGaR 三 seed/full 15k。fixed-split held-out evaluator 已完成：plane
SuGaR 为 24.026 dB / 0.799 SSIM，`manifold_full` 为 19.171 / 0.574；torus
SuGaR 为 20.603 / 0.687，`manifold_full` 为 23.620 / 0.613。

下一次需要你运行的任务将是一个小规模定向消融，而不是 full baseline：

1. plane：提高 self-projected support regularizer，检验能否消除边缘毛刺和竖向拉伸。
2. torus：加入 visibility-aware 低贡献质量清理，检验能否提高 SSIM，同时保持
   normalized varifold 与 certified mass 优势。

plane 的 seed-0 定向配置与自动验收现已冻结。请运行：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_support_sweep_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_support_sweep_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/plane_support_sweep_v1.json
```

这会运行两次 7k（support `0.003` 与 `0.010`），复用旧数据，不会覆盖正式结果。
验收要求相对原 `manifold_full`：PSNR 下降不超过 0.3 dB、SSIM 下降不超过 0.01，
同时 Chamfer 至少改善 5%、normal 至少改善 10%。跑完把最后两行 `PASS/FAIL`
告诉我；未通过不扩三 seed。

状态：**已完成，两档均 FAIL，不要重跑，也不要扩三 seed。** `support=0.010`
只改善 Chamfer 0.27%、normal 0.81%；`support=0.003` 全面退化。下一步由我先把
self-projected support 与真正的 multi-view data anchor 分离，冻结新机制后再给 GPU
命令。

## Action 5：RGB multi-view data anchor（待运行）

真正的 RGB 数据锚点已实现并通过 CPU 梯度测试、20-step 和 100-step GPU smoke。
默认纹理/可见性 mask coverage 约 42%--48%；当前 rasterizer 不提供有效 depth，代码
会显式使用 detached center z-buffer，未静默跳过 loss。

请运行 seed-0 plane 的两档定向验证：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_multiview_anchor_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_multiview_anchor_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/plane_multiview_anchor_v1.json
```

共两次 7k，分别为 `lambda_mv=0.05/0.20`，每 4 步计算一次额外配对视角。
仍使用相同预注册阈值；两档都 FAIL 就停止，不继续扫权重。

状态：**已完成，两档均 FAIL。不要重跑或扩 seed。** 弱档几何改善不足 1% 且
PSNR 下降 0.49 dB，强档全面退化；coverage 非零，失败原因不是实现未启动。
下一步由我实现固定 COLMAP point support，使约束来自输入数据而非模型自生成表面。

## Action 6：固定 COLMAP support（待运行）

固定输入支撑已实现，默认关闭，不影响任何旧结果。单测证明它主要惩罚法向偏移、
保留切向自由度；100-step GPU smoke 中 residual 从 0.0335 降至 0.0274，coverage
约 99%--100%。请运行：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_support_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_support_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/plane_static_support_v1.json
```

共两次 7k，`lambda_static=0.01/0.05`。输出在独立目录，不覆盖原 manifold_full。

状态：已完成。`0.01` 除 normal 改善 7.04% 未达 10% 外，其余指标全部改善；
`0.05` 几何显著改善但 PSNR 下降 0.543 dB。原验收阈值不修改。

## Action 7：固定 support Pareto knee（待运行）

只运行两档之间唯一的预注册插值点 `lambda_static=0.02`：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_support_v2.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_support_v2.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/plane_static_support_v2.json
```

阈值保持不变，不再增加其他权重点。

状态：已完成。几何全部通过，但 PSNR 下降 0.550 dB，故判为 `TRADEOFF`，不是全面失败。

## Action 8：冻结几何的 appearance recovery（待运行）

从 Action 7 的几何 checkpoint 初始化，严格冻结 center、scale、rotation、opacity，
并禁用 densification，只优化 SH/颜色 1000 steps：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_appearance_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_static_appearance_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/plane_static_appearance_v1.json
```

这是一次 1k 短任务。几何指标应仅有 PLY 精度级差异；目标是把 held-out PSNR
恢复到至少 18.871 dB，同时保持现有几何改善。

状态：**已完成，TRADEOFF，不要重跑。** 几何与 Action 7 完全一致，但 held-out PSNR
从 18.621 降至 18.267，属于 3-view appearance overfit。该分支停止。

## 阶段收口

暂不继续追加 GPU 任务。当前应保留 fixed-support Pareto，并设计跨 seed/跨场景的
最小确认协议，而不是继续对 plane seed-0 调参。

## Action 9：fixed support 跨形状迁移（待运行）

不再调整权重。将 plane 上 rendering-balanced 的 `lambda_static=0.01` 原封不动
迁移到 torus seed-0，只运行一次 7k：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/torus_static_support_transfer_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/torus_static_support_transfer_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/torus_static_support_transfer_v1.json
```

若 torus 也通过，下一步才扩 plane/torus seeds 1--2；若失败，则 fixed support
只报告为 plane Pareto 诊断，不升级为默认方法。

状态：已完成，严格 `FAIL`。Normal 改善 27.4%、varifold/coverage/PSNR/SSIM 均
改善，但 Chamfer 恶化 35.9%。拆分显示 completeness 改善而 accuracy 恶化，定位为
局部切平面缺少 trust region 导致的切向无限外推。

## Action 10：torus local-chart trust region（待运行）

权重保持 `0.01`，只补理论上必要的局部图定义域 `tangent_radius_cap=2.0`：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/torus_static_trust_region_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/torus_static_trust_region_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/torus_static_trust_region_v1.json
```

只运行一次 7k；不再扫描 trust-region 半径。

状态：已完成，`NEAR-PASS`。Chamfer 改善 27.57%、normal 改善 9.38%、varifold
改善 11.68%、mass 提升到 66.2%、PSNR guardrail 通过；normal 距门槛 0.62 个
百分点，SSIM 超 guardrail 0.00013。停止调参，转多 seed 验证。

## Action 11：trust-region 多 seed 确认（待运行）

同一参数原封不动运行 plane/torus 的 seeds 1--2，共四次 7k：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/static_trust_region_confirm_v1.json \
  --stage train --resume
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/static_trust_region_confirm_v1.json \
  --stage evaluate --resume
python scripts/summarize_targeted_sweep.py \
  --manifest experiments/manifests/static_trust_region_confirm_v1.json
```

这是统计确认，不再修改权重、trust radius 或验收阈值。

状态：**已完成，0 PASS / 1 TRADEOFF / 3 FAIL。** 不要重跑或继续调 fixed-support
参数。该机制对 sparse COLMAP 初始化具有明显 seed sensitivity，只保留为诊断结果。

## 当前无需执行

停止追加 GPU 实验。下一步由我整理主方法、负结果边界和可发表的 claim，避免继续
围绕单一 synthetic benchmark 调参。

状态：claim/evidence matrix 与论文骨架已完成。下一阶段转 DTU 真实数据。

## Action 12：准备 DTU 真实数据（待执行，不占 GPU）

项目盘空间不足，**不要把数据下载到本项目或 E 盘**。当前实际使用 D 盘：

```bash
mkdir -p /mnt/d/emgs-real/dtu-preprocessed
mkdir -p /mnt/d/emgs-real/dtu-official
mkdir -p /mnt/d/emgs-real/outputs
```

需要两份数据：

1. 2DGS 官方 README 提供的 `DTU+COLMAP (3.5GB)` 预处理数据，解压后应出现
   `/mnt/d/emgs-real/dtu-preprocessed/DTU/scan24` 等目录。该项已完成并验证。
2. DTU 官方 evaluation point clouds，目录需包含 `ObsMask/` 和 `Points/stl/`，放在
   `/mnt/d/emgs-real/dtu-official/`。该项尚未完成。

下载入口记录在 `third_party/2d-gaussian-splatting/README.md` 第 3、148 行。先只确认
pilot scans：`24, 65, 105`。数据就位后执行：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
for scan in 24 65 105; do
  python scripts/prepare_real_scene_split.py \
    --scene /mnt/d/emgs-real/dtu-preprocessed/DTU/scan${scan} --interval 8
done
python scripts/preflight_dtu_real.py \
  --data-root /mnt/d/emgs-real/dtu-preprocessed/DTU \
  --official-root /mnt/d/emgs-real/dtu-official \
  --work-root /mnt/d/emgs-real
```

状态：**已完成，三个 pilot scans 全部 `ready: true`。** 实际官方包保持原始双层
目录，`/tmp/emgs-dtu-official-layout` 用软链接提供标准 `ObsMask/` 与 `Points/stl/`
布局；若系统清理 `/tmp`，需重建软链接。三个重复压缩包已删除，解压数据保留。

当前不要手动训练；下一步由我冻结真实 runner 和 DTU 输出 adapter。

状态：runner、官方 DTU evaluator adapter 和 summary 已冻结。官方 evaluator 已修正
为忽略 macOS `._*` mask，并固定下采样随机种子；距离定义未改变。

## Action 13：DTU scan24 首个真实 pilot（已完成）

先只运行 scan24 的 vanilla 与原始 `manifold_full`，均为半分辨率 7k：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 24 --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 24 --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24
```

输出全部位于 `/mnt/d/emgs-real/outputs/dtu_real_pilot_v1/`。先验证一个真实场景；
scan24 未完成前不要启动 scan65/105、2DGS 或 SuGaR。

首次运行状态：vanilla 已完成；manifold_full 在 iteration 3100 因 8GB OOM 中断。
真实场景 vanilla 最终约 50.1 万 Gaussian。第二次 manifold 已保存 checkpoint：
1000/2000/3000 分别为 8.7万/21.5万/32.8万点，随后在 3100 densify 临时峰值 OOM。
runner 现从 `chkpnt3000` 续跑、在 3000 后停止 densification，并使用
`mcgs_max_points=2048, knn=12, refresh=250`。该结果是 8GB resource-capped
diagnostic，与 vanilla 点数不匹配，不能作为最终公平排名。

最终状态：训练、官方 DTU 几何评测和修正后的 heldout 渲染评测均已完成。
`manifold_full` 相对 vanilla 的 DTU accuracy 改善 10.37%，overall 改善 1.96%，
completeness 退化 1.57%；PSNR 为 30.856 对 30.904，仅低 0.049 dB。完整数字见
`RESULTS-LATEST.md`。当前不要启动 scan65/105：下一步先做同 densification schedule
的 matched-control，解除 50.1 万与 32.8 万 Gaussian 的预算混杂。

## Action 14：DTU scan24 matched-control（已完成）

只新增 `scan24_vanilla_matched`：它与 `manifold_full` 都在 iteration 3000 后停止
densification，但不启用任何 manifold loss。原始两组结果不会覆盖。

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 24 --method vanilla_matched --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 24 --method vanilla_matched --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24
```

输出位于
`/mnt/d/emgs-real/outputs/dtu_real_pilot_v1/scan24_vanilla_matched/`。若再次 OOM，
直接保留最后 30 行日志，不要修改 batch、分辨率或 densification 参数；该对照通常
比原始 vanilla 占用更低。跑完后停下，不要继续 scan65/105。

最终结果：matched vanilla 为 329,604 Gaussians，manifold 为 327,623；两者资源规模
已基本匹配。manifold 的 DTU overall 为 1.7142，对照为 1.7163，仅改善 0.124%；
accuracy 改善 0.43%，completeness 改善 0.012%，PSNR 下降 0.112 dB。原先相对完整
vanilla 的约 10% accuracy 提升主要由提前停止 densification 解释，不能归因于
manifold loss。

## Action 15：DTU scan65 成对复验（已完成）

只运行 matched vanilla 与 manifold，验证 scan24 的弱正向趋势能否跨场景复现：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 65 --method vanilla_matched --method manifold_full \
  --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 65 --method vanilla_matched --method manifold_full \
  --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24 --scan 65
```

两轮都会在 iteration 3000 后停止 densification，并每 1000 iteration 保存 checkpoint。
若 manifold OOM，原命令加 `--resume` 重跑即可；不要改变参数。完成后停下，不运行
scan105。

最终结果：manifold 相对 matched vanilla 的 accuracy 改善 9.32%、overall 改善
2.54%，PSNR 提升 0.624 dB，completeness 轻微退化 0.073%；Gaussian 数差 1.47%。
两场平均 overall 改善 1.33%，当前 decision 为 `INCOMPLETE 2/3`。

## Action 16：DTU scan105 最终成对验证（已完成，FAIL）

这是冻结 replication rule 的最后一场，不再调整 loss、阈值或资源参数：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 105 --method vanilla_matched --method manifold_full \
  --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 105 --method vanilla_matched --method manifold_full \
  --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24 --scan 65 --scan 105
```

完成后查看 summary 最后的 `Decision: PASS/FAIL` 并停下。若 OOM，仅使用同一命令
`--resume`，不要改变参数。按当前两场结果，scan105 仍需满足点数差不超过 5%、
PSNR delta 不低于 -0.3 dB；其 overall 相对改善至少约 0.33% 才能使三场均值达到
1% 门槛。

最终结果：scan105 overall 退化 1.85%，三场 mean overall 仅改善 0.271%，因此冻结
规则判定 **`FAIL (3/3)`**。2/3 正向、PSNR guardrail 和点数 guardrail 均通过，
只有 mean-overall 门槛失败。不要修改阈值或追加同配置 seed 来“救”该判定；当前
RGB-only manifold loss 不足以稳定改善真实场景 coverage。下一步理论/方法工作应
转向外部 data-coercive depth/normal anchor，而不是继续扫描现有 loss 权重。

## Action 17：scan105 COLMAP-anchor 机制诊断（已完成，post-hoc 成功）

只在失败场景 scan105 运行一次 post-hoc diagnostic。该方法在 `manifold_full` 上增加
固定稀疏 COLMAP support，权重固定为 synthetic 已使用的保守档 0.01，不做 DTU
调参。它使用 RGB-SfM 几何，因此不能归入 RGB-only，也不是 GT/sensor depth。

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 105 --method manifold_colmap_anchor \
  --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 105 --method manifold_colmap_anchor \
  --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24 --scan 65 --scan 105
```

该运行每 10 iteration 取最多 2048 个 trainable Gaussians，对确定性采样的 8192 个
固定 COLMAP support 求约束，以限制 8GB 峰值。跑完即停，不调整权重。判断问题只有
一个：scan105 的 completeness/overall 能否相对 `manifold_full` 恢复；无论结果如何，
原 RGB-only `FAIL (3/3)` 保持不变。

最终结果：相对 matched vanilla，accuracy 改善 0.30%、completeness 改善 8.92%、
overall 改善 6.47%，PSNR 下降 0.210 dB，点数差 1.33%；相对 `manifold_full` 的
overall 改善 8.17%。因此该机制诊断成功修复了 scan105 coverage，但原冻结
RGB-only decision 仍应显示 `FAIL (3/3)`。不要把后者误认为 Action 17 失败。

## Action 18：COLMAP-anchor 前瞻性两场复验（已完成，PASS）

scan105 只作为 discovery；在查看其余结果前已冻结 scan24/65 replication rule。权重、
schedule 和资源配置与 Action 17 完全相同，不做场景调参：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_dtu_real_pilot.py \
  --scan 24 --scan 65 --method manifold_colmap_anchor \
  --stage train --execute --resume
python scripts/run_dtu_real_pilot.py \
  --scan 24 --scan 65 --method manifold_colmap_anchor \
  --stage evaluate --execute --resume
python scripts/summarize_dtu_real_pilot.py --scan 24 --scan 65 --scan 105
```

最终输出会同时显示两个互不覆盖的结论：`Frozen RGB-only decision` 应继续为 FAIL；
`Anchor replication decision` 要求 scan24/65 两场 overall 均改善、平均改善至少 1%，
每场 PSNR delta 不低于 -0.3 dB、点数差不超过 5%。完成后停下，不调权重。

最终结果：scan24/65 overall 分别改善 1.62%/4.32%，两场平均改善 2.97%；PSNR
分别提升 0.132/0.297 dB，最大点数差 0.38%。冻结判定为 **`PASS (2/2)`**。
RGB-only `FAIL (3/3)` 仍独立保留。下一项公平性缺口是 SuGaR 尚未在这三个 DTU
场景按同一 split、官方 evaluator 和明确 extractor 口径运行；不能拿 synthetic
SuGaR pilot 与 DTU 3DGS 数字直接比较。

## Action 19：SuGaR-DTU scan105 公平 pilot（部分完成，待 3090 续跑）

使用与 matched 3DGS/anchored method 相同的 scan105、固定 7-view `test.txt` 和
`scan105_vanilla_matched` 7k checkpoint。配置仍是已登记的 8GB pilot：50k SDF
samples、200k mesh vertices、2k refinement，不冒充官方 full-budget。

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
conda activate quad
python scripts/run_sugar_pilot.py \
  --manifest experiments/manifests/sugar_dtu_scan105_pilot.json \
  --execute
conda run --no-capture-output -n sugar python \
  scripts/evaluate_sugar_rendering.py \
  --manifest experiments/manifests/sugar_dtu_scan105_pilot.json
```

大文件通过 `third_party/SuGaR/output/*/scan105` 软链接写入
`/mnt/d/emgs-real/outputs/sugar_dtu_pilot_v1/scan105/sugar_internal/`。稳定结果位于：

- 原生 SuGaR mesh：`scan105/dtu_native_mesh/results.json`；
- refined Gaussians 经统一 extractor：`scan105/dtu_patch_mesh/results.json`；
- 渲染：`scan105/render_metrics.json`。

若训练报错，保留最后 50 行日志并停下；不要改 bbox、SDF samples 或 mesh vertices。
36GB 剩余空间足够本 pilot，但不要同时启动 scan24/65。原生 mesh 与统一 extractor
必须同时报告，不能只挑较好的一项。

2026-07-04 现场状态：coarse 15k 训练已完整保存到
`sugar_internal/coarse/scan105/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`；
随后在 surface level 0.3 生成 10,000,032 个 surface points，进入 foreground
mesh reconstruction 时主机崩溃。当前没有 coarse mesh、refined checkpoint 或任何
`results.json`，因此 Action 19 不得记为完成。换用 3090 后原样重跑上述命令；
runner 会自动传入现有 `--coarse_model_path`，从 mesh extraction 继续，不重训 coarse
model。不改 manifest 的 pilot 预算与评测口径。

最终状态：**已完成（3090）。** 迁移到 autodl 后环境变化：数据根改为
`/root/autodl-tmp/emgs-real`，官方 DTU 包为原始双层目录
（`Points/Points/stl/` 与 `SampleSet/SampleSet/MVS Data/ObsMask/`）。新增
`scripts/dtu_official_layout.py`，在评测前把标准 `ObsMask/` 与 `Points/stl/` 软链接
到 manifest 的 `official_gt_root`（`/tmp/emgs-dtu-official-layout`）；`run_sugar_pilot.py`
现支持从已抽取的 `refined_gaussians.ply`/`coarse_mesh.ply` 续跑 mesh extraction，不重训
coarse。三份结果均已生成：

- 原生 SuGaR mesh：overall `1.2861`（d2s `1.1734` / s2d `1.3988`）；
- refined→统一 extractor：overall `1.5396`（d2s `0.8156` / s2d `2.2636`）；
- 渲染：PSNR `30.728` / SSIM `0.9268`（8 test views）。

对照同 scan105 的 matched 3DGS overall `0.9831`、anchored `0.9195`、PSNR `32.7--32.9`。
即在此 8GB pilot 预算下 SuGaR 明显落后，但这是单场景 discovery、且 SuGaR 非官方
full budget，只作同机同 split 诊断。完整表见 `RESULTS-LATEST.md` 的
“SuGaR-DTU scan105 公平 pilot”。仓库根 `error.txt` 是修复前一次手动评测（`$DTU`
误指向 SampleSet 双层目录、缺 `stl105_total.ply`）的残留日志，问题已由
`dtu_official_layout.py` 解决，可删除。

## Action 20：SuGaR-DTU scan24/65 复验（待执行，需 3090 + sugar 环境）

**目的：**把 SuGaR 的 DTU 公平对照从单场景 scan105 扩到与 3DGS/anchor 相同的三个
scans。scan105 已作为 discovery 完成；scan24/65 用**完全相同**的 8GB pilot 预算与
评测口径复验，使 SuGaR 首次拥有跨场景、同 split、同 extractor 的 DTU 数字。

**前置状态（我已冻结）：**scan24/65 的 `_vanilla_matched` 7k checkpoint 已存在于
`/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1/`；官方 GT 的 `stl024/stl065`
与 `ObsMask24/65` 均在本地；磁盘剩余约 200 GB，足够两场。唯一未冻结项是 SuGaR 前景
`bbox`：scan105 manifest 里的 bbox 是 SuGaR 自身按
`radius = 1.1 × max‖cam_center − mean‖`、`factor=1.0` 自动算出的前景框，写死只为
可复现。scan24/65 的框依赖 SuGaR 归一化后的相机中心，无法用 COLMAP 原始坐标 CPU
直接推出，需按同一公式在 sugar 环境里算一次。

**代码冻结（已完成）：**`run_sugar_pilot.py` 现允许 manifest 省略 `scene_bbox`；
缺省时不传 `--bboxmin/--bboxmax` 并传 `--center_bbox True`，由 SuGaR 用官方默认
`radius = 1.1 × max‖cam_center − mean‖`、`factor=1.0`、以相机平均为中心的前景框，
derivation 完全可复现且跨 scan 一致。scan105 原有显式框路径保持不变。已新增
`experiments/manifests/sugar_dtu_scan24_pilot.json` 与 `sugar_dtu_scan65_pilot.json`
（除 scene / 无 `scene_bbox` 外，逐字复制 scan105 的预算与评测口径）。所有输入已核：
两场 `_vanilla_matched` 7k checkpoint、`stl024/stl065`、`ObsMask24/65`、
`sparse/0/test.txt`（各 7 view）均在本地，磁盘剩余约 200 GB。

**bbox 口径说明（需你知晓）：**scan105 manifest 里那组显式框是**略各向异性**的
（half-width 0.536/0.517/0.480），并非 SuGaR 纯自动公式产出的立方体，其确切来源无法从
仓库复原。因此 scan24/65 改用 SuGaR **官方自动前景框**（立方体、相机平均居中），这是
最可复现、跨 scan 最一致的选择。代价是 scan105 与 24/65 的 bbox derivation 不是逐字
相同（scan105 显式框 vs 24/65 自动框），差异仅为前景裁剪框的 ~10% 各向异性。若你要求
三场严格同法，唯一办法是把 scan105 也改为自动框重跑一次——但那会覆盖已有 scan105
mesh。默认**不重跑 scan105**；如需严格统一请明确告知。

**命令（冻结，需 3090 + `sugar` 环境；两场分别单独跑，勿并行）：**

```bash
cd /root/autodl-tmp/E-Manifold-GS
for scan in 24 65; do
  python scripts/run_sugar_pilot.py \
    --manifest experiments/manifests/sugar_dtu_scan${scan}_pilot.json \
    --execute
  conda run --no-capture-output -n sugar python \
    scripts/evaluate_sugar_rendering.py \
    --manifest experiments/manifests/sugar_dtu_scan${scan}_pilot.json
done
```

scan24/65 尚无 coarse SuGaR 模型，会各自从 `_vanilla_matched` 7k 训练 coarse 15k
再抽 mesh + 2k refinement（比 scan105 的续跑更久）。若某场训练/抽取报错，保留该场
`sugar.log` 最后 50 行并停下，不改 bbox、SDF samples、mesh vertices 或预算。

**验收（冻结）：**两场都必须同时报告原生 mesh 与 patch extractor 两个 overall，加上
render PSNR/SSIM，不得只挑较好一项；与各自 scan 的 matched 3DGS/anchor 数字并列进
`RESULTS-LATEST.md`。结论口径保持“8GB pilot 诊断”，不得升级为“优于官方 SuGaR”。

最终状态：**已完成。** 首跑两场都在 mesh extraction 阶段 CUDA OOM——根因不是显存
不足，而是 runner 里 8GB 时代的 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`
让 PyTorch3D rasterizer 的分配碎片化（实测峰值仅约 3.2 GB）。已从 `run_sugar_pilot.py`
去掉该环境变量，改用默认 allocator；scan24/65 均一次通过，评测口径、分辨率与预算
完全未改，因此对结果无方法学影响。两场 coarse 15k checkpoint 已存在，续跑只做
extraction + 2k refinement + 评测。三场结果（含 Action 19 的 scan105）：

| scan | SuGaR native overall ↓ | SuGaR patch overall ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| 24 | 1.2060 | 1.2619 | 25.176 | 0.8355 |
| 65 | 1.6806 | 1.6179 | 28.427 | 0.9459 |
| 105 | 1.2861 | 1.5396 | 30.728 | 0.9268 |

对照 matched 3DGS 几何 overall（24: 1.716、65: 2.464、105: 0.983）与 PSNR
（24: 30.97、65: 31.50、105: 32.87）：**SuGaR 在 scan24/65 几何更好、scan105 几何更差，
渲染三场全面落后**——是清晰的 mesh-vs-splat 取舍，双向都不能宣称全面胜出。完整分析与
accuracy/completeness 分裂见 `RESULTS-LATEST.md` 的“SuGaR-DTU 三场公平 pilot”。
