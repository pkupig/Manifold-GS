# Baseline 接入状态

更新日期：2026-07-03。论文 claim 的最终口径见 `CLAIM-EVIDENCE-ZH.md`。

本文档区分内部消融、已有源码和可比较复现。只有完成同数据、同相机划分、同训练
预算和同一评测器的条目，才能进入方法优劣结论。

| 方法 | 当前状态 | 可以支持的结论 | 缺口 |
|---|---|---|---|
| vanilla 3DGS | 已集成并已运行 | 可作为内部训练基线 | 仍需与外部方法统一预算 |
| first-order normal supervision | 已集成，历史键名为 `tangent` | 可证明 compatibility 相对一阶监督的增量 | 不是 GeoSplat/2DGS 代理 |
| E-Manifold-GS full | 已集成并已运行 | 可报告当前内部机制与消融结果 | RGB-only 稳定性和更多真实场景仍不足 |
| fixed COLMAP support + trust region | plane/torus 多 seed 诊断已完成 | 可解释 data support 与局部图定义域的作用 | 0 PASS / 1 TRADEOFF / 3 FAIL，不能升级为默认方法 |
| manifold + fixed COLMAP anchor (DTU) | scan105 discovery 后，scan24/65 前瞻性复验 `PASS 2/2` | 相对 matched 3DGS 的 DTU overall 两场平均改善 2.97%，PSNR 不降 | 使用 RGB-SfM 稀疏几何；不是纯 RGB-only，尚缺外部 surface baseline |
| SuGaR | seed-0 plane/torus 8GB pilot 已完成；DTU scan105 runner/preflight 已冻结 | synthetic 上 SuGaR 赢 Chamfer/normal/mesh/plane rendering | DTU pilot 待运行；仍缺 full 1M SDF、15k refinement 和多 seed |
| 2D Gaussian Splatting | 官方 30k、plane/torus、3 seeds 已完成 | 当前注册简单对照中本方法数值更优；plane 上 2DGS 全部 collapse | 仍需真实场景和避免 baseline collapse 的广泛验证 |
| GeoSplat | 截至当前未发现官方公开代码 | 只能做论文层面的概念对照 | 不能伪装成官方复现；如自行实现必须明确标注 |

## 比较协议

外部 baseline 必须满足：

1. 使用完全相同的 train/test 相机和图像。
2. 报告训练预算、densification、初始化和额外先验；不能把 oracle depth 方法与
   sparse-RGB 方法放在同一公平排名中。
3. 将最终几何转换到统一坐标系，用本仓库同一个 Chamfer、normal、coverage、
   kernel-varifold 和 mesh evaluator 评测。
4. 同时报告 PSNR/SSIM，防止用渲染质量换几何指标。
5. 至少运行相同 seeds；失败 run 不能静默删除。

## 当前结论边界

当前可靠结论是：在已有内部简单对照中，可靠或经校准的 depth anchor 加
compatibility 表现最好；compatibility 相对 RGB-only 和一阶 normal supervision
显示出几何收益。在注册的 plane/torus sparse-view 对照中，本方法数值优于官方
2DGS 30k，但 plane 的 2DGS 三个 seed 均 collapse，因此这是窄场景鲁棒性证据，
不是普遍优于 2DGS 或 SOTA。SuGaR pilot 在最近点、法向和 mesh 指标上明显更强，
当前方法仅在 torus 的 varifold/coverage 上显示差异优势；不能声称整体优于 SuGaR
或任何未公开实现的 GeoSplat。

DTU 中 vanilla/matched 3DGS 的几何结果来自统一的本项目 patch-manifold extractor，
不是原生 3DGS mesh。当前没有 SuGaR-DTU 同场景结果，因此 DTU 数字不能与
plane/torus SuGaR pilot 横向排名。

轻量预检命令：

```bash
python scripts/preflight_baselines.py \
  --scene /path/to/colmap_scene \
  --checkpoint /path/to/vanilla_3dgs_output
```

SuGaR 预检只有在 checkpoint 包含
`point_cloud/iteration_7000/point_cloud.ply` 时才会判定 ready；这与其官方默认入口一致。

2DGS 原生 PLY 只保存两个切向 log-scale。统一 Gaussian evaluator 前运行：

```bash
python scripts/convert_2dgs_ply.py \
  --input /path/to/2dgs/point_cloud.ply \
  --output /path/to/2dgs/point_cloud_eval.ply
```

转换器添加的 `scale_2` 是用于恢复 quaternion 第三轴法向的评测厚度，不是 2DGS
学习出的参数，因此不能比较或报告该字段对应的 thinness。
