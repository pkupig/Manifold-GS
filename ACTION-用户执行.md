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

- 自动 edit propagation metric；
- collision candidate 的 coverage/false-surface metric；
- OBJ round-trip 与 source mapping 测试；
- manifest schema 和论文表格生成器。

完成后会把唯一可复制命令、预计时长/显存/磁盘、输出目录和 PASS/FAIL 规则补到本项，
再交给你执行。

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

## Action 19：SuGaR-DTU scan105 公平 pilot（待执行，一轮 SuGaR）

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
