# 项目缺口与收口路线

更新日期：2026-07-04。本文件回答“当前方法已经有什么、还差什么、什么结果才能升级
论文 claim”。长任务的用户执行命令仍统一写入 `ACTION-用户执行.md`。

## 一、当前已经成立的部分

1. **Refinement-conservative measure**
   - appearance opacity 与 geometric mass `q_i` 已分离；
   - clone/split 严格守恒质量、重心和 inherited tangent moment；
   - merge/prune 的守恒范围与扰动界已明确；
   - 实现有单元测试和 prune transport ledger。

2. **Local realizability**
   - covariance、MLS support、normal derivative 和基本形式 compatibility 已闭环；
   - conditioning、alignment、cache drift 与 open-patch rejection 已实现；
   - 理论明确区分 realizability error 与 ground-truth error。

3. **Hybrid asset backbone**
   - 可导出 grouped OBJ/PLY、attached/residual full-attribute Gaussian PLY、source mapping、
     collision candidate 和 manifest；
   - sphere seed-0 导出含 28 个 certified patches，collision gate 拒绝一个尺度异常 patch；
   - 现有输出是可追溯 backbone，不是 production-ready asset。

4. **Evidence boundary**
   - RGB-only 与无 anchor DTU 的失败被保留；
   - 固定 RGB-SfM anchor 在两个 replication 场景上有条件改善 overall 2.97%；
   - 该结果只支持 data-coercivity 机制，不承担 SOTA claim。

## 二、P0：方法闭环缺口

### P0.1 Patch-level identifiability certificate

当前 certification 只回答“局部是否像一个可实现曲面”，不能回答“是否由输入观测支持”。
`patch_0027` 是直接反例：它在训练输出中已经是远离主体的平滑 floater，局部 confidence
中位数为 0.80，因此通过 realizability，却没有 ground-truth 身份。

需要把 asset gate 改为：

```text
asset certified = locally realizable AND observationally supported
```

第一版 observation support 至少包含：

- COLMAP sparse support distance 与局部 support count；
- 被多少训练相机看到、有效视差和投影 footprint；
- 与相机到 first-hit 之间 free space 的冲突；
- 多视 photometric consistency；
- patch 级 evidence score 与明确 reject reason。

第二版再加入 restricted rendering Jacobian/Fisher information 的最小特征值，估计局部
几何方向是否真正被 RGB 约束。没有这一项，项目只能做 realizability-aware extraction，
不能称为 identified asset generation。

2026-07-04 实现进度（仍未完成 P0.1）：

- 新增 `manifoldgs.observation_evidence.v1` 共享 cache，已包含自适应 COLMAP sparse
  distance/support count、训练相机 frustum view count、最大视差和投影 footprint；
- asset exporter 可按 patch 的 sparse-supported fraction、训练视图数和视差门限输出
  evidence 与 reject reason；
- scan105 matched vanilla 7k 的 102,783 个 Gaussians 已生成实测 cache，其中
  50,688（49.32%）通过当前 sparse support rule；

2026-07-06 增量（仍未完成 P0.1）：

- 新增 `compute_visibility_evidence`：以点云自身为遮挡体、按像素分箱求首次命中深度，
  把 view count 从 `frustum_no_occlusion` 升级为遮挡感知的 `first_hit_occlusion`
  （`first_hit_view_count` / `occluded_view_count`）；patch 聚合新增
  `insufficient_first_hit_visibility` 拒绝理由与 `--min-observation-first-hit-views` 门限，
  并有 CPU 单测；
- 多视 photometric consistency 的 CPU 分量已实现（`compute_photometric_evidence`：对
  first-hit 可见点在各视图采样投影像素颜色、Welford 累计跨视图 RGB 方差）；
  `scripts/build_observation_evidence.py --images` 一条命令即可把 `photometric_std` /
  `photometric_view_count` 写进 cache，patch 聚合与 exporter 新增
  `max_photometric_std` / `min_photometric_views` 门限与
  `inconsistent_photometry` / `insufficient_photometric_views` 拒绝理由，已有 CPU 单测；
- 仍缺 free-space conflict（需观测/渲染深度）与 restricted rendering Fisher/Jacobian
  证书（需可微渲染），这两项要 GPU/深度，已写入 `ACTION-用户执行.md` 的 Action A4；
  photometric 在真实 DTU 场景上的实际运行与阈值冻结见 A3。在这两项闭环前不得将当前
  cache 称为完整 identifiability certificate。

2026-07-08 增量（仍未完成 P0.1）：A3 photometric evidence 已在真实 scan105 实跑
（见 `RESULTS-LATEST.md` §4.2）。photometric 一致性阈值改为**每场景相对百分位**口径
（`max_photometric_std_percentile`，避免单场景绝对阈值过拟合），已实现+单测，并已在真实
scan105 hybrid bundle 上端到端生效（402 patches：sparse-support 拒 139、相对 p90 photometric
拒 29、accepted 234，见 `RESULTS-LATEST.md` §4.3）。free-space conflict 与 Fisher/Jacobian
证书仍缺（见 A4）。

2026-07-13 增量（仍未完成 P0.1，但取得关键实证）：用三真实场景的 collision-vs-GT
（§4.4）把 collision patch 标为 GT-floater/GT-clean，实测现有 CPU 观测证据对 floater 的
可分性（`RESULTS-LATEST.md` §4.5）。发现：(1) **first-hit view count 是最强 CPU 判别
信号**（scan24 floater 中位 11 vs clean 34，−2.42σ）；(2) **photometric std 反向**——
floater 更平滑更"光度一致"，正是它骗过 photometric gate 的机制，实证了 P0.1 对 `patch_0027`
的论断；(3) **但无单一 CPU 门限可无损清除**（`min_first_hit_views` 在 scan24 去 63% floater
面积即误伤 clean，scan105 的 floater 更难分）。→ 这从真实数据坐实了**第二版 restricted-
rendering Fisher/Jacobian 证书（GPU/A4）的必要性**：GT-free 的 sparse+photometric 观测闸
对"被相机看到但几何脱离真实表面"的 floater 只有部分区分力。未改任何冻结阈值。

### P0.2 训练期 data evidence 与输出证书一致

当前 fixed COLMAP anchor 是训练 loss，asset exporter 的尺度 gate 是输出 heuristic；两者
尚未共享同一 evidence object。需要统一 support cache，使训练加权、离线认证和最终
manifest 使用相同的 patch evidence 字段，避免“训练说可信、导出又用另一套规则”。

## 三、P0：下游应用缺口

### P0.3 可量化编辑任务

至少建立一个合成有 GT、一个真实人工检查协议：

- 对 selected patch 施加刚性/非刚性 deformation；
- 通过 source mapping 更新 attached Gaussians；
- 测量 target-region edit error、boundary leakage、residual contamination；
- 比较 vanilla nearest attachment、Poisson mesh、SuGaR binding 与本文 certified binding。

只展示 Blender 截图不足以支持 asset claim。

2026-07-07 进度：CPU 侧编辑传播评测已就绪（`manifold_gs/edit_metrics.py` +
`scripts/evaluate_edit_propagation.py`）。给定 patch 选择与刚性/非刚性形变，经 source
mapping 传播后报告 target-region edit error、boundary leakage 与 residual contamination，
并把 certified patch binding 与 nearest-radius baseline 并列对比；exporter 现额外写
`attached_patch_ids` 让 mapping 自描述。合成 GT 单测已确认 certified 零泄漏/零污染、
proximity baseline 跨边界泄漏。仍缺的是合成有 GT 的完整 edit 场景与真实人工检查协议、
以及 Poisson/SuGaR binding 的外部对照运行——需数据/训练，见 `ACTION-用户执行.md` A5/P0.3。
2026-07-10：PASS/FAIL 已冻结（`asset-benchmark/1.0`）——certified 泄漏/污染须为 0 且 baseline
泄漏 `>= 1%`（否则场景不具区分度，标 uninformative）；唯一命令
`scripts/run_asset_benchmark.py --bundle <hybrid_asset>`。真实 scan105 已实跑 PASS
（baseline 泄漏 13.5% vs certified 0）。

2026-07-08 执行：已在 sphere seed-0 backbone 上实跑（CPU）。certified binding 零泄漏/
零污染，nearest-radius baseline 泄漏 12.9%、residual 污染 13.9%。数值见
`RESULTS-LATEST.md` 4.1。

### P0.4 Collision 与 unknown-space 评测

Collision candidate 必须评测：

- supported surface coverage；
- unsupported/floater surface area；
- false collision 与 missed collision；
- simplification 后 Hausdorff/normal error；
- unknown region 是否被错误标为 free space。

当前 relative patch-size gate 只能挡住明显异常 component，是安全护栏而非 identifiability
方法。

2026-07-07 进度：评测代码已就绪（`manifold_gs/collision_metrics.py` +
`scripts/evaluate_collision_candidate.py`），可报告 supported surface coverage、
floater/false surface area、Hausdorff、supported normal error、false/missed collision
（contact-proxy）和 unknown-marked-free 比例，均为 CPU 且有单测。仍缺的是在真实场景上
实际运行（需 GT 表面 npz 与 probe 采样），以及 simplification 前后对照——这部分产大日志、
需数据准备，已列入 `ACTION-用户执行.md` A5/P0.4。
2026-07-10：PASS/FAIL 已冻结——coverage 须在 `<= 3% bbox` 的某 tolerance 达 `>= 0.80`
（1% bbox 常紧于方法自身精度，故按 sweep 自适应读取），有 probes 时 false collision `<= 5%`，
`unknown_marked_free` 只报告不 gate。该线需 GT 表面 npz（Gaussian 坐标系），未给 `--gt`
时自动 skip；GT 对齐仍待做。

2026-07-08 执行+诊断：sphere seed-0 collision 对解析 GT 单 tolerance(1% bbox) coverage
26.84% / false 74.18%，但 tolerance 扫描显示这是**评测口径产物**——candidate→GT 中位误差
0.0329，tolerance 卡在其下；2% bbox 下 coverage 已 72.2% / false 22.2%。circumradius/alpha
bridge 过滤实测无改善，证伪"切向外推 bridge"假设，false-surface 均匀分布。真正杠杆是几何
训练精度(GPU)。CPU 侧已给评测加 `coverage_tolerance_sweep`。数值见 `RESULTS-LATEST.md` 4.1。

### P0.5 Texture/appearance round trip

需要 UV/chart atlas 或逐 patch texture baking，并报告：

- baking reprojection error；
- seam error；
- asset round-trip 后 PSNR/SSIM；
- mesh-only、mesh+attached splats、完整 hybrid asset 的质量/大小/速度 Pareto。

2026-07-07 进度：CPU 度量骨架已就绪（`manifold_gs/texture_metrics.py` +
`scripts/evaluate_texture_roundtrip.py`）。逐 patch 把观测色（默认 Gaussian SH DC，可用
`--evidence` 换成多视 `photometric_mean_color`）烘焙到切平面纹理再采样回来，报告
baking reprojection error/PSNR、texel 填充率与相邻 patch 的 seam error/PSNR；已用合成
纹理单测（分辨率越高误差越小、颜色不一致 seam 增大）。仍缺的是真正的 UV/chart atlas、
经渲染器的 round-trip PSNR/SSIM/LPIPS，以及 mesh-only / mesh+splat / hybrid 的
质量-大小-速度 Pareto——这些需渲染/打包，属 GPU，见 `ACTION-用户执行.md` A5/P0.5。
2026-07-10：PASS/FAIL 已冻结——round-trip PSNR `>= 30 dB` 且 baking excess `<= 0.02`
（绝对 seam 由真实色方差主导，不 gate；单 chart 无边界时 seam gate 自动不适用）。真实
scan105 已实跑 PASS（33.7 dB，excess −0.024）。渲染侧 PSNR/SSIM/LPIPS 与 Pareto 属 P1 扩展，
不在本次冻结必跑范围。

2026-07-08 执行+诊断：sphere seed-0 逐 patch 烘焙 PSNR 36.32 dB，seam PSNR 16.86 dB。
诊断证伪"seam=charting/atlas 缺陷"：跨 patch 邻点对的**原始 SH-DC 颜色** disagreement 已
是 16.68 dB，烘焙后 16.86 dB 基本相同（baking excess −0.006），提高分辨率无改善、bilinear
更差。CPU 侧已给 seam 度量并列 raw-color ceiling。**真实 scan105 验证（§4.3）：**多视
`photometric_mean_color` 相对 SH-DC 把 seam 从 12.50 提到 18.36 dB（+5.86），确认颜色源
是主因；但两者 baked seam 都 ≈ raw ceiling，剩余 seam 是真实跨边界颜色方差，共享 atlas
修不动。→ texture 真实杠杆是颜色源质量，不是 UV atlas。数值见 `RESULTS-LATEST.md` §4.1/§4.3。

## 四、P1：论文证据缺口

### P1.1 真实多场景 asset benchmark

现有 sphere 三 seed 的 mesh Chamfer 均值改善 42.0%，但 paired CI 为 INCONCLUSIVE；
DTU anchor 只有两个正式 replication 场景。至少需要：

- 更多真实场景，覆盖高曲率、薄结构、低纹理和遮挡；
- 每个场景报告 identified coverage 与 rejected mass，而非只报 surviving mesh quality；
- matched Gaussian 数、训练预算、extractor 和 held-out rendering；
- 多 seed 或足够多场景的 paired statistics。

### P1.2 真正的 asset baseline

必须与 SuGaR mesh binding、Poisson extraction 以及至少一个当前 surface/mesh GS 方法在
同一 asset task 上比较。不能继续把内部 `tangent` ablation 当外部 baseline，也不能仅靠
varifold 指标宣称 asset 更好。

2026-07-13 增量：补上**首个真外部 asset baseline = Poisson-from-3DGS**（CPU，见
`RESULTS-LATEST.md` §4.7）。同源定向点做 Poisson watertight，与我们的 collision candidate
同 GT 对比：Poisson 靠 watertight 封闭刷高 coverage（49–76%）但假面 54–98%；我们 coverage
保守（26–41%）而假面仅 1–18%。这是 precision–coverage 取舍的直接外部证据。**仍缺** SuGaR
mesh binding 与一个 surface-GS 方法在 edit/collision/texture 同口径下的对比（GPU，SuGaR
native mesh 已在盘，可直接送同一 collision 线）。

### P1.3 Coverage–precision–rendering 三轴报告

认证方法可以通过拒绝更多区域轻易降低 Chamfer。主表必须同时报告：

- precision：accuracy、normal、unsupported area；
- coverage：completeness、certified mass、identified surface fraction；
- appearance：held-out PSNR/SSIM/LPIPS；
- asset utility：edit/collision/baking 指标。

2026-07-13 增量：三场景（scan24/65/105）主表骨架已拼出 CPU 三轴（识别率/surface area、
collision precision floater+p95+normal、asset-utility edit/texture），见 `RESULTS-LATEST.md`
§4.6。coverage(recall) 待 ObsMask 裁剪（待办 A），appearance held-out 渲染待 GPU；两轴已在
表中标 pending。识别率保守（42–58% patches / 50–62% area），precision 轴清晰区分 scan24
floater 簇（18.3% unsupported area）与干净的 scan65/105（<2%）。

## 五、P2：可延后内容

- GLB/glTF 正式封装、材质规范和引擎插件；
- curve/volume mixed-dimensional 专用下游；
- learned confidence model；
- global topology completion；
- production physics collision certification。

这些方向有价值，但不能早于 P0 identifiability 与 asset benchmark，否则项目会继续扩大
工程面而没有形成可证实的主贡献。

## 六、建议的论文主 claim

若 P0 完成，论文可主张：

> We represent adaptive Gaussian splats as a refinement-conservative geometric
> measure and extract hybrid assets only where local realizability and
> observation-derived identifiability certificates agree.

在此之前，只能主张：

> We provide a conservative geometric representation and a realizability-aware
> asset backbone, while demonstrating that observation support remains necessary
> for ground-truth identification.
