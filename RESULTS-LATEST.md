# 最新实验结果总览

更新时间：2026-07-03。

论文允许使用的 claim 与证据等级统一见 `CLAIM-EVIDENCE-ZH.md`；本文件中的局部
最优数值不得脱离对应协议、seed 和 `PASS/TRADEOFF/FAIL` 状态引用。

## -1. DTU 真实场景 matched diagnostic

结果目录：`/mnt/d/emgs-real/outputs/dtu_real_pilot_v1/`。两组均为半分辨率 7k，
使用相同固定 7-view heldout split、相同 patch-based point-cloud extractor 和官方
DTU evaluator。

| method | Gaussians | Accuracy ↓ | Completeness ↓ | Overall ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 501,230 | 1.0320 | **2.4647** | 1.7484 | **30.904** | **0.939** |
| vanilla, matched densification | 329,604 | 0.9290 | 2.5036 | 1.7163 | **30.967** | 0.935 |
| manifold_full, 8GB cap | 327,623 | **0.9250** | **2.5033** | **1.7142** | 30.856 | **0.935** |

scan65 的成对复验：

| method | Gaussians | Accuracy ↓ | Completeness ↓ | Overall ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| vanilla, matched densification | 81,897 | 1.3727 | **3.5558** | 2.4643 | 31.503 | 0.9705 |
| manifold_full, 8GB cap | 80,695 | **1.2448** | 3.5584 | **2.4016** | **32.127** | **0.9723** |

相对 vanilla，`manifold_full` 的 accuracy 改善 10.37%、overall 改善 1.96%，但
completeness 退化 1.57%；heldout PSNR 仅下降 0.049 dB，SSIM 下降 0.0036。
这说明当前约束在真实场景上可以提高已重建表面的精度，而没有造成明显渲染损失，
但尚未改善 coverage。

matched-control 表明上述大幅 accuracy 改善主要来自 densification schedule，而不是
manifold loss：在相同 iteration 3000 截止和近似相同 Gaussian 数下，manifold 相对
matched vanilla 的 accuracy 改善 0.43%、completeness 改善 0.012%、overall 改善
0.124%，PSNR 下降 0.112 dB。两组 patch certification acceptance 均约 38%，生成
mesh 均无 non-manifold edges。因此仅看 scan24 时，结论只是弱正向几何趋势。

scan65 将该趋势复现并显著放大：accuracy 改善 9.32%、overall 改善 2.54%，PSNR
提升 0.624 dB；completeness 轻微退化 0.073%。两个已完成场景的 mean relative
overall 改善为 1.33%，且两场均为正；前瞻性 replication rule 仍为
`INCOMPLETE (2/3)`，必须完成 scan105 后才能判定。

scan105 最终成对验证：

| method | Gaussians | Accuracy ↓ | Completeness ↓ | Overall ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| vanilla, matched densification | 102,783 | **0.5589** | **1.4073** | **0.9831** | 32.866 | 0.9650 |
| manifold_full, 8GB cap | 102,929 | 0.5801 | 1.4226 | 1.0013 | **33.103** | **0.9653** |

scan105 的 overall 退化 1.85%，accuracy 退化 3.79%，completeness 退化 1.08%，
尽管 PSNR 提升 0.237 dB。三场最终 mean relative overall 仅改善 0.271%，低于冻结
规则的 1% 门槛，因此判定为 **`FAIL (3/3)`**。其余检查均通过：2/3 场景 overall
为正、最差 PSNR delta 为 -0.112 dB、最大点数差为 1.47%。不得通过事后降低阈值
把该结果改写为 PASS。

三场平均 accuracy 相对改善 1.98%，但 completeness 平均退化 0.38%；certification
acceptance ratio 与 matched control 基本持平，accepted geometric mass 则三场都更低。
这支持的机制解释是：内部 realizability 约束可能提高部分已观测表面的 precision，
但没有 data-coercive 信号填补 coverage，因此不能稳定改善 overall reconstruction。

### scan105 COLMAP-anchor post-hoc diagnostic

| method | Gaussians | Accuracy ↓ | Completeness ↓ | Overall ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| vanilla, matched densification | 102,783 | 0.5589 | 1.4073 | 0.9831 | **32.866** | 0.9650 |
| manifold_full | 102,929 | 0.5801 | 1.4226 | 1.0013 | **33.103** | **0.9653** |
| manifold + fixed COLMAP anchor | 101,418 | **0.5573** | **1.2818** | **0.9195** | 32.656 | 0.9650 |

相对 matched vanilla，COLMAP anchor 的 accuracy 改善 0.30%、completeness 改善
8.92%、overall 改善 6.47%，PSNR 下降 0.210 dB，点数差 1.33%。相对
`manifold_full`，overall 改善 8.17%。这直接支持 failure analysis：加入 RGB-SfM
导出的稀疏 data anchor 后，scan105 的 coverage 退化被修复。它是看过 RGB-only FAIL
后进行的单场景 post-hoc mechanism diagnostic，不能修改原 `FAIL (3/3)`，也不能声称
三场稳定或优于外部 SOTA。

### COLMAP-anchor prospective replication

scan105 作为 discovery 排除后，scan24/65 的固定配置复验均通过：

| scan | Overall: matched 3DGS ↓ | Overall: anchored ↓ | Relative gain | PSNR delta |
|---|---:|---:|---:|---:|
| 24 | 1.7163 | **1.6885** | **+1.62%** | +0.132 dB |
| 65 | 2.4643 | **2.3577** | **+4.32%** | +0.297 dB |

两场平均 overall 改善 2.97%，两场均为正，最差 PSNR delta 为 +0.132 dB，最大
Gaussian 数差仅 0.38%，因此冻结的 anchor replication rule 判定 **`PASS (2/2)`**。
加上 discovery scan105，三场 anchored method 相对 matched 3DGS 的 overall 均改善，
但正式 replication 统计只使用 scan24/65。该结论支持 RGB-SfM 稀疏几何锚点与
realizability 约束结合的可复现增量，不属于纯 RGB-only claim。

这里的 matched 3DGS 几何不是原生 3DGS mesh：其 Gaussian PLY 同样经过本项目的
patch-manifold extractor，再进入官方 DTU evaluator。当前 SuGaR 仅有 synthetic
plane/torus pilot，没有 DTU 同场景同 extractor 结果，因此本节不能用于宣称 3DGS 或
本方法优于 SuGaR。

### SuGaR-DTU 三场公平 pilot（Action 19 scan105 + Action 20 scan24/65）

三个 DTU 场景 scan24/65/105，各用同一固定 `test.txt`（24/65 为 7 view、105 为
8 view）、同一 `*_vanilla_matched` 7k checkpoint、同一 SuGaR 8GB pilot 配置
（coarse 15k + surface level 0.3 + 2k refinement，50k SDF samples、200k mesh
vertices，**非官方 full budget**）。SuGaR 同时报告原生 mesh 与经本项目 patch-manifold
extractor 的 refined Gaussians，二者都进入官方 DTU evaluator；渲染在同一 mask 口径下
评测。3DGS 侧数字取自 `dtu_real_pilot_v1/summary.json`。

**几何 DTU Chamfer overall（↓ 更好）：**

| scan | matched 3DGS | manifold+anchor | SuGaR native | SuGaR patch |
|---|---:|---:|---:|---:|
| 24 | 1.7163 | 1.6885 | **1.2060** | 1.2619 |
| 65 | 2.4643 | 2.3577 | 1.6806 | **1.6179** |
| 105 | 0.9831 | **0.9195** | 1.2861 | 1.5396 |

**渲染 held-out（↑ 更好，PSNR dB / SSIM）：**

| scan | matched 3DGS | manifold+anchor | SuGaR |
|---|---:|---:|---:|
| 24 | 30.97 / 0.9349 | **31.10** / 0.9350 | 25.18 / 0.8355 |
| 65 | 31.50 / 0.9705 | **31.80** / 0.9710 | 28.43 / 0.9459 |
| 105 | **32.87** / 0.9650 | 32.66 / 0.9650 | 30.73 / 0.9268 |

结论是**明确的 mesh-vs-splat 取舍，而非某一方全面胜出**：

- **几何**：SuGaR 在 scan24（native 1.206 vs matched 3DGS 1.716，好 29.7%）与
  scan65（patch 1.618 vs 2.464，好 34.3%）上明显优于 3DGS/anchor，但在“容易”场景
  scan105 上明显更差（native 1.286 vs 0.983）。原因是 accuracy/completeness 分裂：
  SuGaR 的 watertight mesh 在难场景大幅改善 completeness（scan24 s2d 1.239 vs 3DGS
  2.504），而 3DGS 在已经很干净的 scan105 上 accuracy 本就很低，SuGaR 的 mesh 承诺
  反而牺牲精度。
- **渲染**：SuGaR 三场 PSNR 全面落后 2.1--5.8 dB，因为它把 splat 承诺到 mesh，牺牲了
  novel-view 合成质量；本项目的 anchored 3DGS 在 24/65 甚至略优于 matched 3DGS。

因此可发表口径为：在同机同 split 的 8GB pilot 下，**anchored 3DGS 在渲染上稳定领先、
在最干净场景 scan105 上几何也领先，而 SuGaR 在更难场景的表面 completeness 上领先**。
由于 SuGaR 使用 8GB pilot 而非官方 full budget，这些是同机诊断，不能宣称本方法或
3DGS 普遍优于官方 SuGaR，也不能反过来宣称 SuGaR 普遍优于本方法。

此前约 9--13 dB 的 heldout PSNR 来自导出渲染未应用 DTU
`alpha_mask` 的评测 bug，已经失效；表中是修复 mask 后重新渲染的有效结果。

## 0. 官方 2DGS 30k 外部对照

汇总文件：

`experiments/external/2dgs_plane_torus/official_30k/summary.json`

协议使用官方 2DGS commit `335ad612`、官方 30k 配置、与本项目相同的
3 train/12 test 划分及 3 seeds。2DGS 原生双尺度 PLY 只通过 adapter 补评测用法向
厚度；中心、切向尺度、opacity 和旋转未修改。

| scene / method | Certified Chamfer | Normal median | Kernel varifold | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| plane / E-Manifold-GS 7k | **0.12601** | **60.86 deg** | **0.18732** | **18.894** | **0.549** |
| plane / 2DGS 30k, all-opaque | 0.22521 | 63.61 deg | 0.77964 | 15.230 | 0.387 |
| torus / E-Manifold-GS 7k | **0.24935** | **41.66 deg** | **0.08438** | **22.565** | **0.549** |
| torus / 2DGS 30k | 0.26305 | 63.79 deg | 0.10918 | 14.922 | 0.330 |

plane 的 2DGS 三个 run 均发生 collapse，最终各自只有 5--15 个 surfel，因此表中
使用 all-opaque 而不是不稳定/空的 certified subset，并把该场景判为 baseline
鲁棒性失败。torus 三个 run 正常保留约 2.8 万 surfel；相对 2DGS，当前方法平均
Chamfer 低 5.2%、法向误差低 34.7%、varifold 低 22.7%，PSNR 高 7.64 dB。

这支持“当前注册的 synthetic sparse-view 简单对照优于官方 2DGS 30k”，但不支持
普遍优于 2DGS 或 SOTA：场景只有 plane/torus，plane baseline collapse，且本方法
自己的 RGB-only registered 总判定仍为 `FAIL`。

## 0.1 SuGaR 8GB pilot

结果目录：`experiments/external/sugar_plane_torus/pilot/`。

这是 seed 0 的 pipeline pilot：复用相同 vanilla 7k checkpoint，coarse 训练到 15k，
使用输入 COLMAP 点云 bbox、50k SDF samples（官方为 1M）和 2k refinement。因此它
不是官方 full-budget headline baseline，但足以检验与 surface-aware mesh baseline
的真实差距。

| scene / method | Chamfer | Normal | Varifold | Cert. mass | Mesh Chamfer | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| plane / E-Manifold-GS full s0 | 0.12371 | 60.83 deg | **0.18364** | **56.2%** | 0.13987 | 19.171 | 0.574 |
| plane / SuGaR pilot | **0.10047** | **47.24 deg** | 0.18792 | 49.6% | **0.12147** | **24.026** | **0.799** |
| torus / E-Manifold-GS full s0 | 0.42273 | 36.62 deg | **0.08619** | **41.1%** | 0.56696 | **23.620** | **0.613** |
| torus / SuGaR pilot | **0.05599** | **15.18 deg** | 0.11786 | 7.2% | **0.05786** | 20.603 | **0.687** |

SuGaR 明显赢 nearest-point、normal 和 coarse-mesh 几何；当前方法不能声称整体优于
SuGaR。另一方面，SuGaR 在 torus 的 certified mass 仅 7.2%，且 normalized
kernel-varifold 反而更差。这表明“高质量局部表面/mesh”不自动等于“稳定的离散
几何测度与 coverage”，是 manifold-conservative/varifold 更可信的差异点。由于
固定 `test.txt` 的 12-view evaluator 已补齐。SuGaR 在 plane 的 PSNR/SSIM 均明显
更高；`manifold_full` 在 torus 的 PSNR 高 3.02 dB、SSIM 仍低 0.075，视觉上对应
主体覆盖更完整同时存在漂浮雾状质量。训练日志中的 50--60 dB 是 train PSNR，不能用于
baseline 比较。当前仍仅一个 seed 且 SuGaR 使用 8GB pilot 配置，不能构成最终统计排名。

## 0.2 Plane support-weight 定向诊断

结果：`experiments/benchmarks/plane_support_sweep_v1/targeted_summary.json`。

| 方法 | Chamfer | Normal | Varifold | Cert. mass | PSNR | SSIM | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| manifold_full (`support=0.001`) | 0.12371 | 60.83 deg | **0.18364** | **56.2%** | **19.171** | **0.574** | reference |
| support=0.003 | 0.12662 | 62.71 deg | 0.18546 | 53.0% | 18.767 | 0.529 | FAIL |
| support=0.010 | **0.12337** | **60.33 deg** | 0.18702 | 54.7% | 19.002 | 0.570 | FAIL |

两档均未通过预注册验收。`0.010` 的 Chamfer/normal 仅改善 0.27%/0.81%，远低于
5%/10% 门槛；`0.003` 全面退化。代码审计确认当前 `mcgs_support` 的 target 来自
当前 Gaussian 点云自身的周期性 MLS 投影，因此它是 self-consistency/proximal
regularizer，不是 data-coercive support anchor。该负结果与理论一致：提高内部
realizability 权重不能消除 sparse-RGB 的深度不可辨识性。

## 0.3 RGB multi-view anchor 定向诊断

结果：`experiments/benchmarks/plane_multiview_anchor_v1/targeted_summary.json`。

| 方法 | Chamfer | Normal | Varifold | Cert. mass | PSNR | SSIM | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| manifold_full | 0.12371 | 60.83 deg | **0.18364** | 56.2% | **19.171** | 0.574 | reference |
| multi-view 0.05 | **0.12292** | **60.65 deg** | 0.18643 | **56.2%** | 18.683 | **0.578** | FAIL |
| multi-view 0.20 | 0.13488 | 61.87 deg | 0.19458 | 50.2% | 18.054 | 0.548 | FAIL |

弱档仅改善 Chamfer 0.64%、normal 0.30%，且 PSNR 下降 0.49 dB；强档全面
退化。训练末期有效 coverage 约 39%--59%，排除 loss 未生效。结论是仅靠 RGB
颜色重投影仍不足以消除此低纹理 sparse-view 深度歧义；按预注册规则停止权重扫描。

## 0.4 固定 COLMAP support 与 appearance recovery

| 方法 | Chamfer | Normal | Varifold | Cert. mass | PSNR | SSIM | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| manifold_full | 0.12371 | 60.83 deg | 0.18364 | 56.2% | 19.171 | 0.574 | reference |
| static 0.01 | 0.11313 | 56.55 deg | 0.17923 | **62.4%** | **19.250** | 0.577 | FAIL: normal threshold |
| static 0.02 | 0.10816 | 46.44 deg | 0.17110 | 61.3% | 18.621 | 0.566 | TRADEOFF: PSNR guardrail |
| static 0.05 | **0.10221** | **34.32 deg** | **0.16291** | 56.7% | 18.628 | **0.588** | TRADEOFF: PSNR guardrail |
| static 0.02 + appearance-only 1k | 0.10816 | 46.44 deg | 0.17110 | 61.3% | 18.267 | 0.573 | TRADEOFF |

固定 COLMAP support 产生了明确且可解释的 geometry/rendering Pareto，而不是全面
失败。appearance-only 前后几何指标完全一致，证明冻结正确；held-out PSNR 继续
下降，说明 3-view SH 微调发生外观过拟合，因此停止该分支，不用训练 PSNR 替代。

### Trust-region 跨 seed 确认

固定 `lambda_static=0.01, tangent_cap=2.0`，不调参运行 plane/torus seeds 1--2：

| run | Chamfer 改善 | Normal 改善 | PSNR guardrail | SSIM guardrail | 判定 |
|---|---:|---:|---|---|---|
| plane s1 | +6.93% | +2.29% | pass | pass | FAIL |
| plane s2 | +11.51% | +13.06% | pass | fail | TRADEOFF |
| torus s1 | -25.02% | +26.37% | pass | fail | FAIL |
| torus s2 | +8.48% | -12.66% | fail | fail | FAIL |

聚合为 `0 PASS / 1 TRADEOFF / 3 FAIL`，平均 Chamfer/normal 改善仅
0.47%/7.26%。因此 local-chart trust region 修复了 torus s0 的切向外推，但固定
COLMAP support 对 sparse/noisy initialization 高度 seed-sensitive，不能作为默认
方法或 headline 改进。它保留为理论边界与失败分析，不再继续调参。

## 1. 最新一轮：结构化深度误差

原始结果目录：

`experiments/benchmarks/plane_structured_depth_1500/runs/`

| 深度误差 | Certified Chamfer | Normal median | Kernel varifold | Shape mismatch | Mesh Chamfer |
|---|---:|---:|---:|---:|---:|
| 2% global scale | 0.04950 | 15.32 deg | 0.08997 | 0.41555 | 0.04169 |
| 2% global bias | 0.04656 | 14.69 deg | 0.08926 | 0.40406 | 0.04144 |
| 2% low-frequency warp | 0.03347 | 26.63 deg | 0.10309 | 0.42783 | 0.01799 |
| 30% dropout (invalid old sampler) | 0.05452 | 55.41 deg | 0.15372 | 0.55033 | 0.04440 |
| combined (invalid old sampler) | 0.05824 | 56.16 deg | 0.14841 | 0.54746 | 0.04215 |

结论：scale/bias 能保持良好法向和 compatibility，但产生整体偏移的错误光滑
曲面。旧 dropout/combined 把 missing zero 混入双线性深度，结果已由下面的
mask-normalized rerun 取代。

Affine calibration 新结果：2% scale 校准后 Chamfer/normal/varifold 为
`0.01700 / 10.14 deg / 0.05863`；2% bias 校准后为
`0.02038 / 11.00 deg / 0.06550`，已基本恢复 exact reference。

修正后的 missing-depth 结果目录：

`experiments/benchmarks/plane_missing_depth_masknorm_1500/runs/`

| 条件 | Chamfer | Normal | Varifold | Certified mass | Mesh Chamfer |
|---|---:|---:|---:|---:|---:|
| 30% dropout | **0.01806** | 11.98 deg | 0.06652 | 72.7% | 0.00892 |
| combined | 0.03456 | 28.63 deg | 0.10205 | 54.0% | 0.01895 |
| combined + affine calibration | 0.03424 | **28.20 deg** | **0.09781** | **57.0%** | **0.01358** |

30% random dropout 在正确 mask normalization 下接近 exact；combined 仍明显优于
RGB-only，affine calibration 进一步改善 varifold、coverage、mesh 和 rendering。

## 2. 深度 iid 噪声稳定区间

原始结果目录：

`experiments/benchmarks/plane_depth_robustness_1500/runs/`

| 相对深度噪声 | Certified Chamfer | Normal median | Kernel varifold | Certified mass |
|---:|---:|---:|---:|---:|
| exact reference | 0.01885 | 8.84 deg | 0.06056 | 77.0% |
| 0.5% | 0.02864 | 17.21 deg | 0.07522 | 76.0% |
| 1% | 0.02645 | 21.20 deg | 0.08452 | 62.6% |
| 2% | 0.03096 | 36.17 deg | 0.11355 | 41.4% |
| 5% | 0.04685 | 54.38 deg | 0.14113 | 25.4% |
| RGB adaptive reference | 0.04537 | 55.95 deg | 0.14614 | 30.7% |

结论：`<=2%` iid depth noise 仍明显优于 RGB-only；`5%` 基本退回 RGB-only。

## 3. Densification 定位

原始结果目录：

`experiments/benchmarks/plane_densification_identifiability_1500/runs/`

| 方法 | Certified Chamfer | Normal median | Kernel varifold | Mesh Chamfer | PSNR |
|---|---:|---:|---:|---:|---:|
| RGB | 0.05946 | 64.91 deg | 0.19335 | 0.07109 | 28.017 |
| RGB + compatibility | 0.05011 | 55.18 deg | 0.15167 | 0.03593 | 27.662 |
| RGB + adaptive compatibility | 0.04537 | 55.95 deg | 0.14614 | 0.02872 | 30.264 |
| exact depth + adaptive compatibility | 0.01885 | 8.84 deg | 0.06056 | 0.00726 | 30.863 |

结论：compatibility 有效但不足以锚定 densification 新增自由度；data-coercive
support anchor 与 compatibility 必须同时存在。

## 4. 正式 7k registered 结果

汇总文件：

`experiments/benchmarks/plane_torus_sparse_v2_shape/summary.json`

状态：`FAIL`。这是当前应公开承认的 RGB-only registered 结论，不得用 oracle
实验替换。主要失败项是 normal 与 normalized kernel-varifold 的注册阈值。

## 4.1 Sphere seed-0 asset-utility CPU pilot（P0.3/P0.4/P0.5，2026-07-08）

首次把已实现的三条 asset-utility CPU 度量**实际运行**在既有 sphere seed-0
backbone（`experiments/analytic_sphere_s0_manifold_full_1500_v3`）上，全部为 CPU、
无 GPU、无训练。结果 JSON 位于该目录的 `asset_eval/`。这是单场景 backbone 诊断，
不是论文级多场景 asset benchmark（后者需真实 DTU + baseline，见 `ACTION-用户执行.md`
A5）。

**前置修复：**旧 `hybrid_asset/asset_mapping.npz` 早于 2026-07-07 新增的
`attached_patch_ids` 字段，texture/edit 评测因此拒绝运行。已用同一 `asset/` 源
（`projected_gaussians.ply` + `patch_mesh.ply` + `patch_mesh_meta.npz` +
`projected_manifold.npz`）经更新后的 exporter 重新导出到 `hybrid_asset_reexport/`，
未覆盖旧 bundle；导出内容除新增字段外与旧 manifest 一致（28 patches、collision 拒绝
patch 27）。

**P0.3 编辑传播**（patch 19，沿 +z 平移 0.1×attached bbox 对角线）：

| 绑定方式 | edited pts | 边界泄漏 leaked pts | residual 污染 fraction |
|---|---:|---:|---:|
| certified patch binding | 145 | **0.0%** | **0.0%** |
| nearest-radius baseline | 145 | 12.9% | 13.9% |

certified binding 零跨边界泄漏、零 residual 污染；proximity baseline 跨边界泄漏。
与合成 GT 单测结论一致，这里是在真实 backbone 上的首个数值确认。

**P0.4 collision candidate**（对解析 GT 球面 20k 点，50k 采样）：单一 tolerance
0.0242（1% bbox）下 coverage 26.84%、false-surface 74.18%。**但这个单点数值有误导性**：
诊断（`asset_eval/` + `evaluate_collision_candidate.py --coverage-sweep`）显示 candidate→GT
距离**中位数 0.0329**，即整张表面均匀地"接近但不精确"，而 tolerance 0.0242 恰好卡在
该误差之下。tolerance 扫描：

| tolerance (bbox 比例) | coverage ↑ | false-surface ↓ |
|---|---:|---:|
| 0.5% (0.0121) | 6.7% | 94.7% |
| 1% (0.0242) | 26.8% | 74.2% |
| 2% (0.0485) | **72.2%** | **22.2%** |
| 3% (0.0727) | 88.0% | 7.6% |
| 5% (0.1212) | 97.5% | 0.2% |

在与方法自身精度匹配的 tolerance（≈2% bbox，约 1.5× 中位误差）下 coverage 已达 72%、
false 降到 22%。**证伪了"切向外推 bridge 三角形制造大量无支撑面"的假设**：circumradius
/alpha bridge 过滤实测对 false-fraction 无改善（最紧档仍 0.63，只是把面积按比例削掉），
因为 false-surface 均匀分布、不是集中在长三角。→ 真正的杠杆是 **certified 几何精度本身**
（需训练/GPU），不是 CPU 三角化 mechanism。

**P0.5 texture round-trip**（逐 patch SH-DC 烘焙，分辨率 32）：per-patch reprojection
PSNR **36.32 dB**；相邻 patch seam PSNR **16.86 dB**。**seam 同样是评测口径产物而非
charting 缺陷**：新增的 raw-ceiling 诊断显示，同一批跨 patch 邻点对的**原始输入颜色**
（完全不烘焙）disagreement PSNR 为 **16.68 dB**，与烘焙后 16.86 dB 基本相同，baking
excess error −0.006（烘焙甚至因平均略优）。提高分辨率无改善，bilinear 反而更差。→ seam
是逐 Gaussian **SH-DC 颜色本身的跨边界方差**，共享 UV/atlas 修不了；该度量本就设计用
多视 `photometric_mean_color`（需 A3 GPU 缓存），SH-DC 只是噪声代理下界。

**小结（含 2026-07-08 弱点诊断）：**编辑绑定给出干净正向证据；collision 74% 与 texture
16.86 dB 两个"弱点"经诊断均为**评测配置产物**，不是 mechanism bug——collision 的 tolerance
比方法精度更紧，texture 用了噪声 SH-DC 颜色。因此"先 CPU 修 mechanism"对这两项不适用：
真正的杠杆是几何训练精度（GPU）与多视 photometric 颜色（A3/GPU）。CPU 侧已把两处评测
改成**不再产生误导数值**：collision 加 coverage-vs-tolerance 扫描并报告精度匹配 tolerance，
texture 在 seam 旁并列 raw-color ceiling。

## 4.2 A3 真实 DTU photometric evidence（scan105，2026-07-08）

在 scan105 `_vanilla_matched` 7k 上实跑 `build_observation_evidence.py --images`（CPU/IO，
读 56 个训练视图，14s 完成）。输出 `experiments/observation_evidence/scan105_photometric.npz`。

- gaussians 102,783；sparse_supported_fraction 0.4932（与既有一致）；
- photometric_multiview_fraction **0.9998**（99.99% 被 ≥2 训练视图 first-hit 看到，
  view count 中位 49），说明 DTU 真实覆盖密集，几乎没有 <2 视图的欠观测点；
- `photometric_std`（跨视图 RGB 方差，[0,1]）分布：

| p1 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.012 | 0.040 | 0.054 | 0.077 | 0.110 | 0.150 | 0.194 | 0.318 | 0.090 | 0.411 |

**阈值冻结（未定，需决策）：**分布无明显 knee，中位 0.077 反映真实 DTU 的视相关外观
（高光/曝光）而非全是 floater。候选一致性门限：p90≈0.15 保留 90%、p75≈0.11 保留 75%。
**但从单场景 scan105 冻结一个绝对全局阈值有单场景过拟合风险**（文档反复告诫）；更稳的做法
是把门限表述为**每场景相对百分位**，或先补 scan24/65 的同一分布再定绝对值。当前不擅自
冻结绝对数值。

**与 texture 弱点的关系：**该 cache 现已提供真实多视 `photometric_mean_color`，正是 §4.1
texture seam 需要的非噪声颜色源。但要验证它能否降低 seam，需要 scan105 的 hybrid asset
bundle（当前只有 sphere 有 bundle），即先在 scan105 上跑一次 asset export，属后续项。

## 4.3 scan105 真实场景 hybrid bundle：P0.1 gate 首次生效 + texture 颜色源验证（2026-07-08）

用既有 scan105 `_vanilla_matched` 投影产物（`asset/` 内 projected_gaussians / patch_mesh /
meta / projected_manifold，2026-07-05 已生成，无需 GPU）导出首个**真实场景 hybrid asset
bundle**，位于 `.../scan105_vanilla_matched/hybrid_asset/`，并首次让 observation gate 端到端
生效（evidence = §4.2 的 photometric cache，索引空间已核对与 patch_meta 一致）。

**P0.1 observation gate 首次真实作用**（402 patches）：

| reject reason | patches |
|---|---:|
| accepted | 234 |
| insufficient_sparse_support | 139 |
| inconsistent_photometry（相对 p90 门限）| 29 |

即新的**每场景相对百分位** photometric 门限在真实 scan105 上确实拒掉 29 个通过了 sparse
support 但跨视图颜色不一致的 patch——机制在真实数据上有效，不再只是合成单测。

**texture seam 的颜色源验证**（解决 §4.1/§4.2 留下的尾巴）：同一 scan105 bundle 分别用
噪声 SH-DC 与真实多视 `photometric_mean_color` 作观测色：

| 颜色源 | per-patch reproj PSNR | seam PSNR | raw-color ceiling | baking excess |
|---|---:|---:|---:|---:|
| SH-DC（噪声代理）| 33.70 | 12.50 | 11.86 | −0.024 |
| 多视 photometric_mean_color | **50.73** | **18.36** | 18.24 | −0.002 |

两点确认了 §4.1 的诊断：(1) 换成真实多视颜色使 seam **+5.86 dB**（12.5→18.36），说明
原"seam 弱"很大程度是 **SH-DC 颜色噪声**而非 charting 缺陷；(2) 但两种颜色源下 baked
seam 都 ≈ raw-color ceiling（18.36 vs 18.24），baking excess ≈0，即剩余 seam 仍是**跨
边界的真实颜色方差**，共享 atlas 仍然基本修不动。→ texture 的真实杠杆是颜色源质量
（多视 > SH-DC），不是 UV atlas；论文口径应如此陈述。

## 4.4 三真实场景 asset benchmark + P0.4 确定性对齐（scan24/65/105，2026-07-13）

将 A5 冻结协议 `asset-benchmark/1.0` 首次跑到**三个真实 DTU 场景**。scan24/65 用与
scan105 相同口径导出 bundle（observation evidence + 相对 p90 photometric gate）。

**edit(P0.3) / texture(P0.5) 三场景全 PASS**（表：`experiments/asset_table.md`）：

| 场景 | certified 泄漏 | baseline 泄漏 | 泄漏减 | 往返 PSNR | baking excess | overall |
|---|---:|---:|---:|---:|---:|:--:|
| scan24 | 0 | 0.1474 | 0.1474 | 30.11 dB | −0.054 | PASS |
| scan65 | 0 | 0.2814 | 0.2814 | 35.34 dB | −0.016 | PASS |
| scan105 | 0 | 0.1351 | 0.1351 | 33.70 dB | −0.024 | PASS |

**P0.4 坐标对齐已解决且确定性**：DTU 官方 stl（mm，DTU 世界帧）→ Gaussian/重建帧的变换
就是预处理 `cameras.npz` 里的 `scale_mat`（均匀缩放+平移相似变换），**无需 ICP**。三场景
重建 mesh → 变换后 stl 最近邻中位残差均 **<0.1% bbox**（scan105 0.04%）。均匀缩放保方向，
stl 自带法线直接复用。

**collision 精度（与裁剪无关的方向：候选→GT）：**

| 场景 | floater%(>1%bbox) | 候选→GT 中位 | 候选→GT p95 | 法线中位° |
|---|---:|---:|---:|---:|
| scan24 | **18.25** | 0.121% | **5.13%** | 49.5 |
| scan65 | 0.87 | 0.126% | 0.657% | 51.7 |
| scan105 | 1.54 | 0.058% | 0.501% | 52.2 |

- scan65/105 的 collision candidate 几何忠实（floater <2%、p95 <0.7% bbox）。
- **scan24 是发现**：中位仅 0.12%（主体贴合 GT），但 p95 5.1% + floater 18% —— 非全局失配，
  而是一**簇整片悬浮的 patch**（26/188/143/130/27…，各 100% 面 >1% bbox，离 GT 中位 2–6% bbox）。
  这些 patch 直径 1.5–2.5× 中位，**低于 3× scale gate、且通过 observation gate**，故三道
  GT-free 导出闸都没挡住。→ 诚实结论：**"被相机看到但几何脱离真实表面"的 floater，GT-free
  自证闸挡不住，正需 P0.4 collision-vs-GT（需 GT）才照得出**；这既是 P0.4 指标的价值证明，
  也是 exporter 当前 3× scale gate 对 scan24 偏松的证据（不擅自改冻结阈值，波及 65/105）。
- 法线中位 ~50°（无向）三场景一致，是 splat 派生 patch mesh 高频法线噪声，观察项非 gate。

**待冻结（决策：保留、暂不做）**：`coverage`(recall) 与 `hausdorff` 现被 DTU 背景底盘污染
（完整 stl_total），需官方 **ObsMask + Plane 裁剪**才可比可入 gate；未裁前 benchmark 的
collision 线保持 `skip`，不接入 overall。

## 4.5 P0.1 实证：CPU 观测证据能否分离 GT 坐实的 floater（2026-07-13）

用 §4.4 的 collision-vs-GT 把每个 collision patch 标为 **GT-floater**（>90% 面积 >1% bbox 离 GT）
或 **GT-clean**（<10%），再回看其 per-patch 观测证据（导出时未被 gate 的字段）。样本：
scan24 22 floater / 163、scan105 3 / 215、scan65 0 / 153。

**分布级签名（scan24，floater 中位 vs clean 中位）：**

| 证据字段 | floater | clean | 分离 |
|---|---:|---:|---:|
| median_first_hit_view_count | 11 | 34 | **−2.42σ** |
| median_max_parallax_deg | 78.8 | 99.9 | −1.36σ |
| median_photometric_std | 0.073 | 0.115 | −0.77σ（**反向**）|
| sparse_supported_fraction | 0.60 | 0.65 | −0.34σ |
| median_projection_radius_px | 15.0 | 10.9 | +0.08σ |

两个要点：(1) **first-hit view count 是最强 CPU 判别信号**（floater 被更少相机首次命中，
11 vs 34）；(2) **photometric std 反向**——floater 反而更"光度一致"（平滑 floater），
正是它骗过 photometric gate 的原因，印证 P0.1 对 `patch_0027` 的论断，并解释了导出的
sparse+photometric 双闸为何漏网。

**但单门限无法无损清除：** 扫 `min_first_hit_views`——scan24 取 15 去掉 ~63% floater
面积却误伤 2 个 clean、取 20 误伤 12 个；scan105 的 3 个 floater 门限 ≤15 一个都去不掉、
=20 才去 1 个却误伤 5 个 clean；scan65（0 floater）门限 ≥20 纯误杀。**即现有 CPU 证据
部分相关但不可分**。

**结论（P0.1）**：GT-free 的 sparse+photometric 观测闸对"被相机看到但几何脱离真实表面"
的 floater 只有部分区分力，无单一 CPU 门限可无损清除 → **实证了需要第二版 restricted-
rendering Fisher/Jacobian 几何可识别性证书**（GPU，见 `ACTION-用户执行.md` A4）。未改任何
冻结阈值；这是诊断，不是 gate 变更。

## 4.6 P1.3 三轴主表骨架（scan24/65/105，CPU 部分，2026-07-13）

把前几节散落的结果收敛成论文 P1.3 主表形态。识别、precision、asset-utility 三轴已有真实
CPU 数；coverage(recall) 待 ObsMask 裁剪（待办 A），appearance 待 GPU held-out 渲染。

| 轴 | 指标 | scan24 | scan65 | scan105 |
|---|---|---|---|---|
| 识别 | patches identified% | 48.3% | 42.1% | 58.2% |
| 识别 | identified surface area% | 54.2% | 49.5% | 61.9% |
| 识别 | rejected: sparse / photo | 175 / 22 | 206 / 20 | 139 / 29 |
| precision | collision floater%（unsupported area）| **18.25%** | 0.87% | 1.54% |
| precision | candidate→GT p95（%bbox）| 5.13% | 0.66% | 0.50% |
| precision | normal median（°）| 49.5 | 51.7 | 52.2 |
| asset-util | edit leak reduction | 0.147 | 0.281 | 0.135 |
| asset-util | texture round-trip PSNR | 30.1 dB | 35.3 dB | 33.7 dB |
| coverage(recall) | completeness vs GT | pending A | pending A | pending A |
| appearance | held-out PSNR/SSIM/LPIPS | pending GPU | pending GPU | pending GPU |

**读法：**(1) 认证是保守的——三场景只识别 42–58% 的 patch（surface area 50–62%），其余按
sparse/photometric 证据拒绝，符合"realizability-aware backbone"的保守口径。(2) precision 轴
把 scan24 的 floater 簇（18.25% unsupported area）与干净的 scan65/105（<2%）清晰分开，正是
P1.3 要求的"不能靠拒绝更多区域来刷 Chamfer"——precision 与识别率同表暴露质量差异。(3) 两轴
pending 已标注，不含糊。这张表对应论文可主张的当前口径（`PROJECT-GAPS-ZH.md` 六）：保守几何
表示 + realizability-aware backbone，且实证 observation support 仍不足以完全 GT 识别。

## 4.7 P1.2 首个外部 asset baseline：Poisson-from-3DGS（CPU，2026-07-13）

给 asset 任务补第一个**真外部 baseline**（此前 edit/texture 的 baseline 是内部"朴素分离
度"代理，P1.2 明确不认）。选 **Poisson-from-3DGS**：对**同源**定向点 `projected_points.ply`
（95k，与我们 patch mesh 同一批点、但**未经认证/观测门限**）做 Poisson watertight
extraction（open3d depth=9 + 密度分位 0.1 修剪 + 输入 bbox 裁剪，标准实践配置），再用
§4.4 的同一 collision-vs-GT 指标与我们的 collision candidate 同场景对比。

加了两个外部 baseline：**Poisson-from-3DGS**（同源定向点 `projected_points.ply`，open3d
depth=9 + 密度分位 0.1 + 输入 bbox 裁剪）与**SuGaR native mesh**（已发表 surface-GS 方法，
用其 DTU 评测的 `culled_mesh.ply`，DTU mm 帧经 `scale_mat_inv` 转回 Gaussian 帧）。三方同
场景、同 GT、同 collision-vs-GT 口径对比：

| 场景 | 方法 | 面数 | floater%（假面/unsupported area）| coverage@1% |
|---|---|---:|---:|---:|
| scan24 | **ours** | 6.6k | **18.3%** | 37.1% |
| scan24 | sugar-culled | 26.6k | 78.7% | 69.8% |
| scan24 | poisson | 17.6k | 98.5% | 51.5% |
| scan65 | **ours** | 5.5k | **0.9%** | 26.3% |
| scan65 | sugar-culled | 15.9k | 17.4% | 46.3% |
| scan65 | poisson | 16.8k | 97.4% | 49.0% |
| scan105 | **ours** | 11.2k | **1.5%** | 41.0% |
| scan105 | sugar-culled | 79.0k | 7.7% | 51.9% |
| scan105 | poisson | 68.3k | 54.1% | 75.9% |

**结论（P1.2 / P1.3 precision–coverage 取舍的外部证据）：** 我们的观测认证候选在**全部
三场景 floater% 最低**（0.9–18.3%），甚至我们最脏的 scan24（18%）也远优于 SuGaR(78.7%)/
Poisson(98.5%)。SuGaR（有 watertight regularization 的已发表方法）介于朴素 Poisson 与我们
之间（7.7–78.7%），符合预期。代价是我们 coverage 最保守（26–41% vs SuGaR/Poisson 46–76%）：
两个 baseline 靠 watertight 封闭刷高 coverage，但把大量扫描仪未见区域封成**假碰撞面**。用于
physics/collision，假面即错误碰撞——这正是本方法的价值主张：**用 coverage 换诚实 precision，
而非靠拒绝刷 Chamfer**，且对手含一个真正的已发表方法。

口径说明：precision 主指标用稳健的**面积比 floater%**（Poisson 的 candidate→GT p95 距离被
`projected_points` 远端 floater 撑爆，属离群伪影）。SuGaR-culled 是 DTU-mask 过的物体级
mesh（对 SuGaR 更有利，非稻草人）。edit/texture 线的外部 baseline（需把对手 mesh 绑回
Gaussian 做编辑传播）与 2DGS 仍待补，见 `ACTION-用户执行.md`。

## 4.8 P1.2 edit 轴外部对比：结构化可编辑性（2026-07-13）

edit 轴的外部对比有个陷阱：现有 edit 指标的 `edit_region` **由 patch 定义**，certified
binding 对它天然零泄漏，直接拿去比外部方法是**循环论证**。诚实、非循环的外部论点应落在
**结构本身**：watertight 抽取能否提供结构化的编辑边界。

对每个方法数连通分量（可编辑单元的天然边界）：

| 场景 | ours 认证 patch 数 | 单 patch 中位面积占比 | SuGaR 最大连通分量 | Poisson 最大连通分量 | proximity 基线泄漏 |
|---|---:|---:|---:|---:|---:|
| scan24 | 381 | 0.08% | 99.0% | 97.5% | 14.7% |
| scan65 | 390 | 0.12% | 99.5% | 99.7% | 28.1% |
| scan105 | 402 | 0.10% | 98.3% | 99.1% | 13.5% |

**结论**：Poisson/SuGaR 的 watertight mesh 把物体、底座、背景**焊成单一连通体**（最大分量
占 97.5–99.5% 三角形），因此基于 mesh 连通性的区域编辑**没有任何结构化边界**——要编辑子
区域只能强加一个几何 proximity 切割，而那正是 §E1/4.4 里**泄漏 13.5–28.1%** 的 radius
baseline。相反，认证 patch 把同一表面分解成 **381–402 个观测认证的独立编辑单元**（每个中位
仅 0.1% 面积），配合 certified binding 实现**零泄漏**的结构化编辑。这与 collision 轴（§4.7）
一起，构成 asset-utility 相对 watertight 抽取的两条外部证据：**碰撞诚实（不幻想假面）+ 编辑
结构化（有认证边界，不被迫用会泄漏的 proximity）**。

## 5. 当前结论边界

- 已证明/验证：显式几何质量、守恒 refinement、局部 cross-field realizability
  约束、cache/MLS 稳定条件、data-identifiability 分解。
- 已观察：compatibility 在无 densification 和有可靠 support anchor 时显著改善
  normal/varifold；结构化 depth bias 会形成错误但可实现的曲面。
- 已验证（asset-utility，三真实场景 scan24/65/105）：certified 编辑绑定零泄漏（baseline
  0.135–0.281）、texture 往返 30–35 dB，冻结协议 `asset-benchmark/1.0` 全 PASS（§4.4）。
  P0.4 揭示 scan24 存在 GT-free 闸挡不住的 floater 簇——collision-vs-GT 的独立价值。
- 尚未证明：RGB-only 普遍优于 2DGS、SuGaR 或任何 GeoSplat 实现；真实深度先验下的多 seed 收益；完整
  Gauss-Codazzi training；任意场景的 manifold/GT 收敛。
