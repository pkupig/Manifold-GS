# 论文叙事骨架

## 一句话问题

几何感知 GS 即使拥有薄 splat、法向和曲率，也未必形成对 refinement 稳定的几何
测度；局部几何场可实现，也未必能由 sparse observations 唯一识别为 GT 曲面。

## 核心主线

1. **表示层**：从 Gaussian covariance 得到 tangent/dimension，但用独立 `q_i`
   表示几何质量，避免把 opacity 当面积。
2. **守恒层**：定义 clone/split/merge/prune transport，使自适应 refinement 保持
   零阶质量，并在适用算子中保持一阶矩与切向矩。
3. **可实现性层**：用局部 MLS chart 与基本形式 compatibility 检测独立学习的
   support/tangent/curvature 是否可能来自同一曲面。
4. **可辨识性层**：证明并实验区分 realizability 与 data coercivity；Bonnet 条件
   不能代替观测证据。
5. **输出层**：只对 confidence-certified 区域投影和建 open patches，拒绝把未知
   区域伪造成 watertight topology。

## 建议章节

### 1. Introduction

- 现有方法主要优化 primitive surface-likeness 或 mesh quality。
- 缺口不是“还少一个 normal loss”，而是 refinement law、集合级 measure 与
  realizability/identifiability 边界。
- 贡献只采用 `CLAIM-EVIDENCE-ZH.md` 中允许的表述。

### 2. Related Work

- 2DGS：局部 disk primitive。
- SuGaR/mesh GS：surface alignment、mesh extraction、asset binding。
- GeoSplat：manifold/varifold、曲率与高阶几何，明确承认宽泛概念重合。
- Point-cloud varifold、MLS、Gauss–Codazzi/Bonnet：数学工具来源。

### 3. Refinement-Conservative Gaussian Measure

- 定义 `V_G = sum_i q_i delta_(mu_i,P_i)`。
- opacity 与 `q_i` 分离。
- split/merge/prune conservation propositions 与 bounded-Lipschitz stability。

### 4. Confidence-Certified Realizability

- covariance spectrum typing 与局部 chart。
- tangent、shape、symmetry/Gauss compatibility。
- cache drift、Gram conditioning、reject region。
- 明确：这些条件不蕴含 GT identifiability。

### 5. Identifiability Analysis

- RGB image Jacobian 的条件 coercivity。
- visible depth graph 的稳定性结果。
- unseen/unmatched mass residual。
- 为什么低 photometric loss、正确 Bonnet residual 和错误曲面可以同时成立。

### 6. Experiments

- A：守恒算子与 refinement invariance，作为最强机制实验。
- B：analytic identifiability ladder（RGB、depth、noise、bias、calibration）。
- C：registered RGB-only 失败，作为理论边界验证。
- D：2DGS 与 SuGaR 外部对照，严格标注 narrow/pilot。
- E：fixed-support/trust-region 多 seed 负结果，证明 sparse/noisy anchor 不稳定。
- F：asset-utility CPU 度量（编辑传播、collision coverage/false-surface-vs-tolerance、
  texture round-trip/seam）在 sphere 与真实 scan105 backbone 上的首批数字；含观测
  identifiability gate（sparse support + multi-view photometric，每场景相对百分位门限）
  在 scan105 上端到端拒绝 patch 的统计。强调 collision/texture 的表观弱点为评测口径产物。
- 后续必须新增真实数据、经渲染器 round-trip 与外部 binding 对照，当前结果不得写成 SOTA 表。

### 7. Limitations

- RGB-only 无普遍 identifiability。
- 局部 chart 不保证全局拓扑。
- sparse COLMAP support 对 seed/noise 敏感。
- SuGaR full budget、GeoSplat implementation 与真实 benchmark 尚缺。
- asset-utility 的表观弱点（collision false-surface、texture seam）经诊断为评测口径产物，
  真实杠杆是几何精度与多视颜色质量；UV atlas 与经渲染器 round-trip 仍缺。

## 图表优先级

1. **主理论图**：appearance opacity 与 conserved geometric mass 的分离，以及
   split 前后相同离散测度。
2. **逻辑图**：data evidence → identifiability；compatibility → realizability；两者
   缺一不可。
3. **成功/失败并列图**：oracle depth 成功、RGB/fixed-support 失败，展示相同低
   compatibility residual 不代表 GT 正确。
4. **外部对照 Pareto 图**：Chamfer/normal 与 varifold/coverage/rendering 分轴，
   不制造单一“总体最好”排名。

## 当前投稿判断

理论对象和失败边界具有真实研究价值，但实验尚不足以支持 oral。下一决策点不是
继续 synthetic 调参，而是：守恒机制能否在标准真实数据上提供稳定、不可被普通
regularizer 替代的收益。
