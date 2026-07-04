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

## 5. 当前结论边界

- 已证明/验证：显式几何质量、守恒 refinement、局部 cross-field realizability
  约束、cache/MLS 稳定条件、data-identifiability 分解。
- 已观察：compatibility 在无 densification 和有可靠 support anchor 时显著改善
  normal/varifold；结构化 depth bias 会形成错误但可实现的曲面。
- 尚未证明：RGB-only 普遍优于 2DGS、SuGaR 或任何 GeoSplat 实现；真实深度先验下的多 seed 收益；完整
  Gauss-Codazzi training；任意场景的 manifold/GT 收敛。
