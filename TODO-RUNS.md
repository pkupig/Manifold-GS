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
- [ ] 在已有保质量 representation pruning 基础上，增加显式 semantic deletion ledger；
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

- [ ] 按冻结参数运行 `experiments/manifests/plane_torus_sparse_v1.json`；plane 与
  torus 分场景报告 paired CI，不把两个不同难度的场景只汇总成一个数字。

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
