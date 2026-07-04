# Claim–Evidence Matrix（论文唯一口径）

更新日期：2026-07-03。本文档优先级高于早期 proposal、TODO 和可行性文档。

## 证据等级

- `PROVEN`：有明确假设、命题和证明边界，并有对应单元测试验证实现。
- `IMPLEMENTED`：代码路径和诊断可运行，但不等价于方法有效或普遍收敛。
- `EMPIRICAL-CONDITIONAL`：只在已注明的数据、先验、seed 或协议下成立。
- `UNSUPPORTED`：当前证据不能支持，论文不得使用肯定句。

## 可用 Claims

| Claim | 等级 | 直接证据 | 论文边界 |
|---|---|---|---|
| 显式几何质量 `q_i` 可与 opacity 分离，并在 clone/split 中守恒 | `PROVEN` | `geometric_measure.py`、守恒测试、prune ledger | 不代表渲染 opacity 或真实面积自动正确 |
| 保质量、重心与切向矩的 refinement 对离散 varifold 扰动有条件上界 | `PROVEN` | `THEORY-PROOF-SKETCH.md`、`THEORY-STABILITY.md` | 要求采样、切向误差和局部尺度条件 |
| covariance spectrum 可统一生成维度标签、切平面、质量与局部图诊断 | `IMPLEMENTED` | diagnostics、projection、patch mesh、统一 evaluator | 分类阈值不是语义 GT，也无拓扑保证 |
| certified patch 可导出为带 source 映射的 hybrid asset bundle | `IMPLEMENTED` | `asset_bundle.py`、导出脚本、smoke test、sphere seed-0 bundle | 仅为 asset backbone；尚无 UV/GLB、编辑任务或生产级 collision 验证 |
| cross-field 基本形式残差用于检测局部 realizability，而不只是一阶法向一致性 | `PROVEN/IMPLEMENTED` | `THEORY-BONNET.md`、compatibility cache 与测试 | Bonnet 条件不选择 GT，也不保证优化器找到正确曲面 |
| reliable depth/data anchor 与 compatibility 结合时可显著改善 analytic geometry | `EMPIRICAL-CONDITIONAL` | oracle、noise、affine calibration、densification 实验 | oracle 不是公平 sparse-RGB 方法；结构化 bias 可得到错误但可实现曲面 |
| RGB-only 下 realizability 与 identifiability 必须分开 | `PROVEN + EMPIRICAL` | Jacobian/coercivity 分析；RGB、multi-view、fixed-support 失败 | 这是问题分解与负结果，不是 SOTA reconstruction claim |
| 当前简单 analytic 协议中优于官方 2DGS 30k | `EMPIRICAL-CONDITIONAL` | plane/torus 3 seeds 同 split 结果 | plane 的 2DGS collapse；不得外推到真实场景或普遍排名 |

## 禁止 Claims

| 不可用表述 | 原因 |
|---|---|
| “首次把 manifold/varifold 用于 GS” | GeoSplat、topology/geometry-aware GS 和相关工作已占据宽泛表述 |
| “首次将 varifold 用于 3DGS” | 缺少全球优先权证据，且 GeoSplat 已直接重合 |
| “理论保证实验一定 PASS” | 定理只给条件 realizability/稳定性，不保证 GT identifiability 或优化成功 |
| “整体超过 SuGaR/GeoSplat” | SuGaR pilot 明显赢 Chamfer、normal、mesh；GeoSplat 无同协议公开实现 |
| “RGB-only 方法稳定恢复正确流形” | registered 7k 总判定为 FAIL；fixed-support 多 seed 为 0 PASS/1 TRADEOFF/3 FAIL |
| “已经具备 oral 证据” | 缺真实 benchmark、完整外部 baseline、多场景稳定主结果与关键消融闭环 |

## 当前贡献表述

建议摘要使用以下窄而真实的版本：

> We formulate adaptive Gaussian splats as a refinement-conservative discrete
> geometric measure, separate geometric quadrature mass from appearance
> opacity, and introduce confidence-certified local realizability diagnostics
> based on tangent and fundamental-form compatibility. Our analysis separates
> representational realizability from data identifiability; controlled
> experiments show that compatibility is effective with coercive geometric
> evidence, while sparse RGB alone does not reliably select the ground-truth
> surface.

中文含义：贡献是“守恒表示 + 可实现性诊断 + identifiability 边界”，不是宣称已解决
sparse-view surface reconstruction。

## 实验证据总账

| 证据 | 状态 | 能说明什么 |
|---|---|---|
| registered RGB-only plane/torus 7k | `FAIL` | compatibility 单独不足以保证 GT 几何 |
| exact/noisy/calibrated depth ladder | 有条件成功 | 数据 coercivity 是缺失环节；低噪声与校准有效 |
| fixed RGB multi-view reprojection | `FAIL` | 低纹理颜色一致性不足以提供稳定深度约束 |
| fixed COLMAP support + trust region 多 seed | `0 PASS / 1 TRADEOFF / 3 FAIL` | 局部图定义域必要，但 sparse/noisy support 仍 seed-sensitive |
| official 2DGS 30k | 窄协议数值领先 | 只支持 analytic simple-comparison claim |
| SuGaR 8GB pilot | 混合结果 | SuGaR 赢局部/mesh 几何；本方法在部分 varifold/coverage/rendering 指标有差异 |
| GeoSplat | 无可运行官方对照 | 仅允许概念与论文层面对照 |
| DTU matched 3DGS | **`FAIL 3/3`** | 两场正、一场负；mean overall 仅改善 0.271% < 1%，PSNR/点数 guardrail 通过 |
| DTU scan105 + fixed COLMAP anchor | post-hoc 单场景成功 | 相对 matched 3DGS overall 改善 6.47%、completeness 改善 8.92%，PSNR -0.210 dB；不可外推 |
| DTU fixed COLMAP anchor replication | **`PASS 2/2`** | discovery 排除；scan24/65 overall 均改善，平均 +2.97%，PSNR/点数 guardrail 通过 |

COLMAP-anchor 的 scan105 是 discovery，不进入 replication 判定。在查看 scan24/65
anchor 结果前冻结两场复验规则：两场 overall 都必须为正、平均相对改善至少 1%，
每场 PSNR delta 不低于 -0.3 dB、Gaussian 数差不超过 5%。PASS 仅支持“RGB-SfM
稀疏锚点机制可复现”，不恢复 RGB-only claim，也不支持 SOTA。

最终 replication 判定为 PASS。允许表述为：“在固定 RGB-SfM 稀疏 support 下，所提
realizability pipeline 相对 matched 3DGS 在两个前瞻性 DTU 场景上稳定改善官方
overall metric，平均相对改善 2.97%，且不损失 heldout PSNR。”不得省略 RGB-SfM
anchor、两场景和 matched resource schedule 限定。

DTU 的前瞻性判定规则在查看 scan24 后、运行 scan65/105 前冻结，因此只可称为
“replication rule”，不可追溯声称为 scan24 的预注册。三场景需满足：mean relative
overall 改善至少 1%，至少 2/3 场景改善，每场 PSNR delta 不低于 -0.3 dB，且成对
Gaussian 数差不超过 5%。PASS 只支持真实场景增量有效，不支持 SOTA。

最终结果未通过该规则。三场 accuracy 平均相对改善 1.98%，completeness 平均退化
0.38%，说明当前 loss 更接近局部 precision regularizer，而不是稳定的 surface
coverage/reconstruction 改进。论文可报告这一机制性负结果，不得声称真实 DTU 几何
优于 matched 3DGS。

## 投稿成熟度

当前适合继续发展为“理论与诊断型论文”，但尚不具备 oral 级实验证据。进入主会强
claim 前至少需要：标准真实数据、统一预算外部 baseline、多 seed 稳定结果，以及
证明守恒/compatibility 对最终任务有不可替代增量的消融。
