# TODO Runs

## 当前状态：暂时不要跑 2k

此前给出的 `synthetic_mcgs_2k` / `synthetic_vanilla_2k` 只能继续验证工程链路，
不能验证论文假设。原因是旧 synthetic 的 RGB 与正弦曲面点云没有共同的成像
GT。现在继续跑它会消耗 GPU，但不会产生可用于论文的结论。

当前没有需要用户接手的 GPU 任务。等下面的 analytic GT 与 evaluator 完成后，
再明确通知用户接手。

几何离线管线现已闭合并可回写/重载 Gaussian PLY，详见
`GEOMETRY-PIPELINE-STATUS.md`。目前仍不交接长训练：硬投影虽改善 GT 几何，
但即时渲染下降约 2.4 dB，下一步先改成 proximal/渐进投影。

Fundamental-form compatibility 离线诊断、可微局部导数缓存和训练调度已完成，
并通过解析正确场、破坏场、normal-sign invariance 及梯度测试。单个解析 sphere
的 300-step 机制实验表明：相对 tangent-only，延迟启用 symmetry 可同时改善
Chamfer、normal、varifold 和 symmetry residual，PSNR 变化仅 -0.018 dB。详见
`FUNDAMENTAL-COMPATIBILITY.md`。这仍不是论文结论，下一步必须进入多 seed、
有 densification 的正式 manifest。

正式 7k seed-0 首跑发现并修复两项短跑无法发现的问题：densification/pruning
改变点数后旧图索引会越界，现在点数变化会强制重建图；upstream opacity
pruning 会删除约 50% 的显式 `q`，现在 `--mcgs_preserve_pruned_mass` 会把被删
质量最近邻转移到保留表示，并累计运输代价。700-step densification smoke 中总
质量保持为 `6.19273`，相对解析 GT 面积误差保持 `0.57%`。该传输严格守恒零阶
质量，一阶矩误差由质量加权运输距离控制，尚未实现严格一阶守恒。

解析数据现在把 train/held-out 相机都写入 COLMAP，并由 `test.txt` 明确划分。
旧 300-step 表的 PSNR 仅为 train PSNR。正式 held-out 指标由
`scripts/evaluate_rendered_images.py` 计算。第一版协议、执行器和汇总器分别是
`experiments/manifests/sphere_sparse_v1.json`、
`scripts/run_experiment_manifest.py`、`scripts/summarize_experiment_manifest.py`。

7k pilot 进一步发现旧 symmetry 可通过降低 chart coverage 规避约束。修正版已将
二阶导数算子放回实际 center support，对全部候选施加置信度下限权重，并加入弱
support proximal。1500-step densification calibration 相对 tangent-only：certified
mass coverage 61.0%→69.3%，certified Chamfer 0.07165→0.05438，symmetry
0.17586→0.11182；面采样 patch-mesh Chamfer 0.15229→0.13278，法向
27.77°→23.69°，非流形边为 0。当前 patch mesh 仍有 30 个分量和 464 条边界，
Poisson fallback 会生成大面积无支撑伪表面，不能作为主输出。

修订后的正式协议为 `experiments/manifests/sphere_sparse_v2.json`。它把可认证
几何层与 radiance residual 层分开报告，并把 coverage、held-out rendering 和
面采样 mesh 指标同时列为判据。

v2 的 7k seed-0 已闭环。full 相对 vanilla：certified Chamfer 改善 37.1%，
normal 改善 62.1%，robust varifold 改善 17.1%，mesh Chamfer 改善 26.9%，mesh
normal 改善 30.5%；held-out PSNR/SSIM 均高于 vanilla。相对 tangent，full 的
certified Chamfer/normal/varifold、symmetry、coverage 和 mesh 也全部更好。
唯一未过预注册门槛的是 varifold：17.1% < 20%。这是单 seed pilot，不能据此
修改阈值；下一步直接跑 seed 1/2，汇总 paired CI。

asset 层新增守恒 robust quadrature：accepted 与 residual 总质量分别严格不变，
accepted q 向截断 kNN 面积松弛。7k full 的 certified varifold 从 0.1178 降至
0.0777。patch triangulation 同样截断连接半径，full mesh Chamfer 从 0.1517
降至 0.1169。当前仍是 70 个局部分量、1002 条边界，定位为 editable surface
backbone，不声称 watertight。

- [ ] 当训练内 proximal、守恒 refinement、asset 输出闭环或最终 idea 定型时，
  新增一份中文 `FRAMEWORK-中文.md`，完整说明问题、理论对象、算法、训练、
  mesh/asset 输出、实验判据及与 GeoSplat 等方法的区别。

## 论文主线的验证对象

完整定义见 `THEORY-VALIDATION.md`。每次实验必须同时声明：

1. **假设**：H1 measure consistency、H2 refinement invariance、H3 sparse-view
   geometry、H4 topology/asset 或 H5 mixed dimension；
2. **GT**：与 RGB/深度/法向由同一个解析曲面和相机生成；
3. **对照**：至少 vanilla、thin-only、完整方法；
4. **主指标**：varifold distance / GT Chamfer / normal / topology；
5. **保护指标**：held-out PSNR/SSIM/LPIPS；
6. **判据**：预先固定阈值、至少三个 seed，不能只看最好的一次。

`thinness_median`、`r23_median` 和 `surface_kept` 只说明优化器做了什么，
不能说明几何正确。

## Codex 当前实现任务

- [x] 用同一解析几何生成 RGB、mask、depth、normal、相机、COLMAP 初始化点；
- [x] 首先支持 plane、sphere、torus；
- [x] 导出 dense GT surface samples、normals、area weights、reference mesh；
- [x] 实现 checkpoint 对 GT 的 accuracy/completeness/Chamfer/normal evaluator；
- [x] 实现 normalized kernel-varifold distance；
- [x] 实现显式质量的 total-mass/first-moment evaluator；
- [x] 把几何质量 `q_i` 从渲染 opacity 中分离；
- [x] 实现严格守恒的离散测度 split/merge 算子和单元测试；
- [x] 将守恒质量传输接入官方 3DGS clone/split densification；
- [x] 在已有保质量 representation pruning 基础上，增加显式 semantic deletion ledger；
- [x] 生成固定的多 seed 实验 manifest、自动 PASS/FAIL 汇总和 GPU 命令；
- [x] 实现 Gaussian→quadratic MLS→confidence graph→patch mesh→projected Gaussian PLY；
- [x] 支持 `--mcgs_initial_ply` 从投影后的表示继续训练；
- [x] 实现 differentiable fundamental-form symmetry/Gauss loss；
- [x] 实现二阶 compatibility 的独立 start/ramp 调度；
- [x] 修复 densification 后图缓存 stale index；
- [x] 实现 pruning 的零阶保质量最近邻传输和运输代价记录；
- [x] 实现显式 held-out 划分及无外部模型权重的 PSNR/SSIM evaluator；
- [x] 实现 certified/radiance 双层评估、守恒 robust quadrature 与 coverage guardrail；
- [x] 实现面采样 mesh evaluator 和稳健连接半径截断；
- [x] 完成 sphere_sparse_v2 的 seeds 0/1/2 全方法 paired CI；

## sphere_sparse_v2 三种子正式结论

预注册 manifest 的 seeds 0/1/2 已全部闭环，机器可读结果位于
`experiments/benchmarks/sphere_sparse_v2/summary.json`。总体判定为
`INCONCLUSIVE`，不是 `PASS`，也没有任何指标触发有统计把握的 `FAIL`。

| 三种子均值 | vanilla | tangent | manifold full |
|---|---:|---:|---:|
| certified mass coverage | 54.95% | 50.71% | **63.44%** |
| certified Chamfer | 0.24090 | 0.22158 | **0.15211** |
| certified normal | 42.53 deg | 32.45 deg | **27.47 deg** |
| robust varifold | 0.09124 | 0.08821 | **0.08047** |
| sampled mesh Chamfer | 0.28287 | 0.26534 | **0.16346** |
| sampled mesh normal | 44.89 deg | 41.78 deg | **36.08 deg** |
| held-out PSNR | **21.4959** | 21.4666 | 21.3542 |
| held-out SSIM | **0.5539** | 0.5514 | 0.5395 |

相对 vanilla，full 的均值改善为 point Chamfer 37.1%、normal 35.4%、
varifold 11.7%、mesh Chamfer 42.0% 和 mesh normal 19.4%；coverage 增加
8.49 个百分点。held-out rendering 的均值代价为 0.142 dB PSNR 和 0.0144
SSIM。由于只有三个 seed，所有主要 improvement 和 rendering guardrail 的
95% paired t-CI 都跨过判定边界，因此不能写成统计显著。

seed 2 暴露了目前最重要的限制：full 的 certified Chamfer 和 mesh 指标仍优于
vanilla，但 certified normal 为 42.67 deg，几乎没有超过 tangent 的 42.94 deg。
这表明 compatibility 当前稳定改善 support/asset，却尚未稳定辨识 normal
orientation。下一轮应增加 seed 数并加入 plane/torus；若只在 sphere 上调权重，
会把方法调成单场景结果。中文 framework 文档继续等 topology 路线和跨场景验证
确定后再定稿。

- [x] 按冻结参数运行 `experiments/manifests/plane_torus_sparse_v1.json`；plane 与
  torus 分场景报告 paired CI，不把两个不同难度的场景只汇总成一个数字。

`plane_torus_sparse_v1` 已完成，结论为 `FAIL`。直接原因不是 pruning 或 held-out
渲染崩坏，而是 compatibility 训练链路当时缺了理论对象里的 `shape match`
主项：训练内只优化了 `support / tangent / symmetry / gauss`，没有直接优化
`A_pred - A_support`。现在 `manifold_gs/training_hooks.py` 已新增
`--mcgs_lambda_shape`，并把 `gauss` 比较改到 shape operator 层。下一轮协议为
`experiments/manifests/plane_torus_sparse_v2_shape.json`，用更晚启用、较弱 symmetry
和显式 shape-match 做复验。

理论稳定性重构已开始，见 `THEORY-STABILITY.md`。当前已证明局部 graph 模型下
`tangent/value + shape` 对“到可实现曲面集合的距离”具有 coercivity，并给出当前
Gaussian position/tangent kernel-varifold 的 coupling 上界。非线性命题要求额外的
chart 条件 `|n_cov dot n_support| >= gamma`；现有 MLS confidence 只衡量 support
planarity / sampling anisotropy。现已统一训练 cache、offline compatibility 和 MLS
projection 的 radius-normalized design，并在 compatibility 输出中增加
`normal_alignment_abs_{p10,median}`、`linear_gram_min_{p10,median}` 和
`quadratic_gram_min_{p10,median}`；全局尺度 `1e-3 / 1 / 1e3` 不变性测试已通过。
下一步是把 angle/Gram margin 接入训练期 reject/weight 规则，并补 center
errors-in-variables 与 kNN neighbor stability 界。现已加入默认关闭的
`--mcgs_compatibility_alignment_floor` / `--mcgs_compatibility_gram_floor`，并证明
`d_{k+1}-d_k>4 delta` 时 kNN 集合不变；offline 输出新增 normalized kNN gap 与
PCA normal eigengap。下一步先从已有 checkpoint 重算这些 margin 以冻结阈值，再
启用 gating 跑新实验，不再把 symmetry、Gauss、Codazzi 单独当作能推出 GT
正确性的约束。

Seed-0 旧 checkpoint 的只读 margin 重算表明：plane/torus 的
`knn_gap_ratio_p10` 都只有约 `0.0014--0.0017`，所以严格 kNN 不变条件只允许
约 `0.00035--0.00043` 个 neighborhood radius 的相对中心漂移；固定 100-step
cache 很可能超出该范围。angle/Gram 的低分位在不同场景间相差明显，暂不冻结
全局 hard threshold。训练现已记录 `mcgs_cache_drift_{mean,max}`，并支持默认关闭的
drift-triggered refresh；下一步应用短机制校准测 drift 时间序列，而不是直接重跑
完整 7k manifest。

300-step sphere drift 校准已完成。固定 100-step cache 的 max drift 为 `0.10924`，
`0.01` adaptive trigger 将其压到 `0.00996`，refresh `3 -> 31`，训练耗时约增加
27%。adaptive 使 symmetry `0.21058 -> 0.18650`、shape mismatch
`0.23274 -> 0.22991`、Codazzi `0.42354 -> 0.41432`，但 varifold 只从
`0.096276 -> 0.096048`，normal 从 `9.49 deg` 退化到 `10.03 deg`。结论：stale
cache 是 compatibility estimator 的真实问题，但不是 GT FAIL 的主因；暂不直接
用 `0.01` 跑 7k，下一理论重点转向 realizable surface 内的数据可辨识性约束。

数据可辨识性理论现已分解为两个可检验命题：对单视图可见 depth graph，`H1`
depth error 可控制 position+tangent 进而控制 kernel-varifold；RGB-only 则必须假设
去除 appearance gauge 后 multi-view rendering Jacobian 的最小奇异值 `beta>0`。
现有 analytic 数据已经有 train-view depth/normal/mask，但当前训练未加载它们。
`plane_torus_sparse_v2_shape` 的场景内相关性也显示 RGB 不是稳定几何代理：torus
的 PSNR-varifold Spearman 为 `-0.067`，plane 为反方向的 `+0.633`。下一实验应先做
RGB / mask-free-space / exact-depth-oracle 与是否加 compatibility 的诊断 ladder；
oracle 只用于定位 H3b，不作为 sparse-RGB 方法结果。

全局可辨识性还必须报告 train-camera union 的 GT visible-area coverage 与 model
visible-mass coverage。理论上 normalized kernel-varifold 满足
`MMD_global <= (1-eta) MMD_visible + 2 eta`；mask 只约束 silhouette 外部，depth
只额外约束 first hit 之前的 free space，二者都不能排除 first hit 后面的重复层。
因此下一步先实现 analytic GT visibility coverage evaluator，再决定 depth-oracle
训练约束，不能用“3 views”替代真实 coverage。

`scripts/evaluate_visibility_coverage.py` 已实现并接入 manifest evaluate stage。
三 seed 结果：plane train-view union GT mass coverage 在所有容差均为 `100%`；
torus 在 bbox depth tolerance `0.5% / 1%` 下约为 `49--50% / 53--54%`。因此
torus FAIL 含有明确的 unseen-surface 不可辨识性，不能期待 compatibility 从 RGB
恢复被遮挡的一半曲面；plane 已完全 first-hit covered，后续应在 plane 上优先审计
photometric Jacobian/appearance gauge，并用 exact-depth oracle 判断 optimizer 与 RGB
证据谁是瓶颈。

300-step plane oracle ladder 已完成（无 densification）：RGB 的 certified
Chamfer/normal/varifold 为 `0.03091 / 63.65 / 0.17021`；point-depth oracle 为
`0.02785 / 63.71 / 0.16908`；compatibility 为
`0.03089 / 23.33 / 0.11478`；oracle+compatibility 为
`0.02768 / 24.65 / 0.11731`。结论：pointwise depth 只明显改善 support position，
compatibility 在 densification 前确实大幅改善 tangent/varifold。7k plane 退化应
定位到 densification、graph/cache 变化或后期 schedule，而不能再简单归因于 RGB
不可辨识。

1500-step densification 定位已完成。RGB / fixed compatibility / adaptive
compatibility / oracle+adaptive 的 certified Chamfer 分别为
`0.05946 / 0.05011 / 0.04537 / 0.01885`，normal 为
`64.91 / 55.18 / 55.95 / 8.84 deg`，varifold 为
`0.19335 / 0.15167 / 0.14614 / 0.06056`。adaptive 将 max cache drift 从
`0.5623` 压到 `0.00996`，改善 symmetry、support、mesh 和 PSNR，但 RGB-only
normal 仍未恢复；oracle+adaptive 同时显著恢复所有几何量。训练轨迹显示
densification 后 RGB-only shape term 升到 `1.3--1.6`，oracle 下保持约
`0.5--0.7`。结论：stale cache 是次要问题，主因是 refinement 增加自由度后缺少
data-coercive support anchor。下一步不是继续加 compatibility 权重，而是测试带噪
depth/normal/free-space 约束，确认从 exact oracle 到现实几何先验的退化曲线。

1500-step noisy-depth sweep 已完成。噪声 `0.5% / 1% / 2% / 5%` 的
Chamfer 为 `0.02864 / 0.02645 / 0.03096 / 0.04685`，normal 为
`17.21 / 21.20 / 36.17 / 54.38 deg`，varifold 为
`0.07522 / 0.08452 / 0.11355 / 0.14113`。`<=2%` 仍明显优于 RGB adaptive
baseline (`0.04537 / 55.95 / 0.14614`)；`5%` 基本退回 baseline。certified mass
随噪声从 exact `77.0%` 降至 `25.4%`，是最敏感 guardrail。下一步现实路线应以
约 `1%` 深度精度为目标，并测试 structured scale/bias、边缘错误和 dropout；不能
把 iid Gaussian noise 结果直接等同于 monocular depth。

Structured depth audit 已完成。2% scale/bias 的 normal 仍为
`15.32 / 14.69 deg`，shape mismatch 为 `0.41555 / 0.40406`，但 Chamfer
`0.04950 / 0.04656` 和 mesh Chamfer `0.04169 / 0.04144` 已略差于 RGB adaptive
support，验证“错误但可实现曲面”null space。2% low-frequency warp 仍有部分收益
(`0.03347 / 26.63 / 0.10309`)；30% dropout 与 combined 基本退回或差于 RGB
baseline。下一实现优先级：用 SfM/初始 centers 对每视图 depth 做 affine
scale+shift calibration，并把 missing-depth coverage 显式并入权重/拒绝规则；不能
依赖 compatibility 自己纠正 coherent depth bias。

已实现基于固定初始 SfM centers 的 per-view robust affine depth calibration：首次
加载深度时拟合 `z_center ~= a * depth_prior + b`，迭代裁掉大残差，之后冻结
`a,b`。2% scale-error 的 10-step 接口校验得到三个视图约
`a=0.961--0.966, b=0.036--0.048`，组合映射在实际 depth range 内恢复正确标尺；
不能单看 `a`，因为窄 depth range 下 scale/shift 高相关。下一轮比较 calibrated
scale/bias 与未校准结果，并测试 combined error 中只能消除 affine 分量的边界。

Affine calibration 1500-step 结果已完成：2% scale 从
`0.04950 / 15.32 / 0.08997` 恢复到 `0.01700 / 10.14 / 0.05863`；2% bias 从
`0.04656 / 14.69 / 0.08926` 恢复到 `0.02038 / 11.00 / 0.06550`，接近 exact。
同时发现旧 dropout/combined 的 bilinear sampler 会把 missing zero 混成浅深度；
已改成 validity-normalized interpolation，旧两项结果作废，只重跑三个修正版。

Mask-normalized rerun 已完成。30% dropout 的 Chamfer/normal/varifold 为
`0.01806 / 11.98 / 0.06652`，接近 exact，证明旧失败完全由 zero-depth 插值污染
造成。Combined 修复后为 `0.03456 / 28.63 / 0.10205`；再加 affine calibration
为 `0.03424 / 28.20 / 0.09781`，coverage `54.0% -> 57.0%`，mesh Chamfer
`0.01895 -> 0.01358`，PSNR `30.47 -> 31.20`。结论：随机 missing 本身在三视图
冗余下可承受；剩余瓶颈是 low-frequency/iid structured residual，而非 dropout。
下一阶段应转向真实或模拟 monocular confidence/edge-correlated error，不再扫描
独立像素 dropout。

训练输出目录现在会额外写出 `geom_mass_prune_ledger.json`，记录每次 pruning 的：

- iteration；
- pruned points / pruned mass；
- 是否启用保质量传输；
- transport cost；
- 分原因统计：`low_opacity`、`big_screen_radius`、`big_world_scale`。

长跑交接命令（支持中断后原命令恢复）：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/run_experiment_manifest.py \
  --manifest experiments/manifests/plane_torus_sparse_v1.json \
  --stage all --resume
python scripts/summarize_experiment_manifest.py \
  --manifest experiments/manifests/plane_torus_sparse_v1.json
```

这里的 GT 是生成器同源写出的解析 surface samples/normals/area weights 与 reference
mesh；RGB、mask、depth、normal 和相机均来自同一 mesh。最终应得到
`experiments/benchmarks/plane_torus_sparse_v1/summary.json`，其中既有 pooled checks，
也有 `scene_results.plane` 和 `scene_results.torus`。理论上希望看到：plane 的
symmetry/normal 不退化，证明二阶项不会凭空制造曲率；torus 的 Chamfer、normal、
varifold 和 mesh 指标优于 tangent，证明贡献不只来自一阶薄化；两者 held-out
PSNR/SSIM 不越 guardrail。若 plane 好而 torus 失败，说明当前离散 compatibility
还不能稳定处理变号高斯曲率；若两者都只改善 Chamfer 而不改善 normal，则主张应
收缩为 conservative asset support，而不能声称恢复了完整 fundamental forms。

已完成部分的入口：

```bash
python scripts/generate_analytic_scene.py \
  --out experiments/analytic_sphere_s0 \
  --scene sphere --train-views 3 --heldout-views 12 \
  --init-points 256 --init-noise 0.01 --seed 0

python scripts/evaluate_geometry_gt.py \
  --ply path/to/point_cloud.ply \
  --gt experiments/analytic_sphere_s0/gt/surface.npz \
  --out path/to/geometry_metrics.json
```

当前 kernel-varifold 会把两边归一化为单位质量，因此只检验 support+tangent，
不能冒充总面积守恒。总面积、first moment 与 pruning transport 由独立 `q_i`
指标另行审计；robust quadrature 保持 certified/residual 两层总质量分别不变。

## 用户下一次接手时应看到的内容

不会再是“观察有没有变好”，而是一个 manifest，例如：

```text
hypothesis: H1 + H3
scene: analytic_sphere
train_views: 3
heldout_views: 20
seeds: [0, 1, 2]
gt_surface: experiments/analytic_sphere/gt/surface.npz
gt_mesh: experiments/analytic_sphere/gt/mesh.ply
methods: [vanilla, thin_only, manifold_full]
primary: [kernel_varifold, chamfer_l1, normal_median_deg]
guardrail: heldout_psnr >= vanilla_psnr - 0.3 dB
pass: varifold -20%, chamfer -15%, normal -20%
```

跑完后 evaluator 应直接输出 `PASS/FAIL/INCONCLUSIVE` 以及置信区间，而不是
让用户凭 summary 猜结论。

## 仍可使用的 smoke test

旧命令只用于检查 CUDA、扩展和训练 hook 是否损坏：

```bash
cd /mnt/e/Users/DELL/Desktop/geo/E-Manifold-GS
python scripts/preflight_3dgs.py
```

已完成的 50 iteration 结果只记录为：训练成功、loss 可导、spectrum 发生移动。
它不进入论文实验表。
