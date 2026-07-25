# Restricted-rendering Fisher 证书协议 v1

状态：**冻结于 2026-07-25，三场正式 sweep 已完成。** 本文是 P0.1/A4 的唯一口径；任何参数、
阈值或场景改动都必须新建 v2，不能覆盖 v1 结果。

## 目标

对已经通过 realizability + sparse/photometric observation gate 的每个 patch，测其**沿局部
法向移动**是否会在已观测训练图像中产生稳定的 RGB 变化。证书只说明“固定外观场条件下的
局部几何可辨性”，不证明 RGB-only 的全局唯一重建。

## 固定输入

- 场景：DTU `scan24 / scan65 / scan105`；输入为各自 `*_vanilla_matched` 的 7k checkpoint。
- 候选：当前 exporter observation gate 接受的 patch；不得事后按 GT floater 标签筛选。
- 视图：该 patch 的 `first_hit_view_count` 对应训练视图；少于 3 个 first-hit 视图直接
  `insufficient_views`，不计算 Fisher。
- 像素：每视图 patch 投影 footprint 内、alpha mask 有效且 patch 为 first-hit 的像素；每 patch
  每视图以固定 seed 0 最多均匀采样 4096 像素。
- appearance gauge：冻结 SH 系数、opacity、scale、rotation 与所有非目标 primitive；只扰动
  该 patch 绑定 Gaussian 的中心。因而本证书明确是 **restricted/fixed-appearance** Fisher。

## 固定扰动与统计量

- 扰动方向：patch PCA/三角网格一致朝向的单位法向；符号无关。
- 步长：`epsilon = 0.005 × scene_bbox_diagonal`，使用中心差分 `+epsilon` 与 `-epsilon`。
- 每个采样像素的导数：`J = (RGB(+epsilon)-RGB(-epsilon)) / (2 epsilon)`；RGB 是线性 [0,1]
  渲染值，按三个通道平方和累积。
- Fisher 标量：所有有效像素的 `mean(||J||²)`；同时报告有效视图数、有效像素数、p10/p50/p90。
- 数值拒绝：任一渲染出现 NaN/Inf、有效像素少于 256，或可见视图少于 3，则该 patch 标
  `numerically_unresolved`，不能记为低 Fisher 或已识别。

## 冻结判定

每场景独立以该场景所有数值有效 patch 的 **Fisher p10** 为阈值（只排除数值无效/视图不足
patch，绝不按 GT 或结果标签排除）。patch 在 `>= p10` 时为 `fisher_supported`，否则为
`weakly_identified`。这是一条保守的场景内排序证书，不能作为跨场景绝对量纲排名。

## 固定产物与报告

每场写 `<bundle>/asset_eval/restricted_fisher_v1.json`，包含 protocol version、checkpoint hash、
patch id、Fisher、有效像素/视图、阈值、状态及渲染随机种子。三场汇总必须同时报告：

1. accepted patch 中 `fisher_supported` / `weakly_identified` / unresolved 的比例；
2. 与已有 collision GT floater/clean 的**只读诊断**（不回调阈值）；
3. 指定 coverage、precision、appearance 三轴表中的变化；
4. GPU 时间、峰值显存和失败日志。

## 禁止事项

不得扫描 epsilon、patch/视图选择、阈值百分位或出现失败后的渲染参数；不得用 GT 标签确定
阈值；不得将已完成的 v1 称为完整的全局 identifiability certificate；它只覆盖固定外观下的局部法向敏感度。
