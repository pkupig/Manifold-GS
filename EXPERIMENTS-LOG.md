# 实验记录（EXPERIMENTS-LOG）

本文件是**实验台账**：每条实验记录「目的 / 命令 / 关键结果 / 结论 / 结果文件路径」，
便于复现与追溯。详细分析叙事见 `RESULTS-LATEST.md`；协议冻结与缺口见
`PROJECT-GAPS-ZH.md` 与 `ACTION-用户执行.md`。

- 数据/产物约定：仓库只存代码与文档，`experiments/` 与 `emgs-real/outputs/` 下的
  bundle、npz、mesh、json 等**数据产物不入 git**（`experiments/` 已 gitignore，
  `emgs-real/` 在仓库外）。本台账用绝对路径引用这些结果文件。
- 运行环境：conda env `sugar`（本机唯一可用环境）。
- 场景：DTU `scan24 / scan65 / scan105` 的 `*_vanilla_matched` 7k checkpoint。
- 路径简写：`OUT = /root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1`，
  `SUG = /root/autodl-tmp/emgs-real/outputs/sugar_dtu_pilot_v1`，
  `DTU = /root/autodl-tmp/emgs-real/dtu-preprocessed/DTU`，
  官方 GT stl = `/root/autodl-tmp/emgs-real/dtu-official/Points/Points/stl/stlNNN_total.ply`。

---

## E1 · A5 asset benchmark 协议冻结 + 三真实场景（P0.3/P0.5）

- **目的**：把 asset-utility benchmark 冻结成唯一命令 + PASS/FAIL，并跑到三真实场景。
- **协议**：`asset-benchmark/1.0`，阈值唯一真源 `manifold_gs/asset_benchmark.py`。
- **命令**（每个场景）：
  ```bash
  # 先导出 bundle（复刻 scan105 口径：observation evidence + 相对 p90 photometric gate）
  python scripts/build_observation_evidence.py --gaussians <asset>/projected_gaussians.ply \
    --colmap-points $DTU/<scan>/sparse/0/points3D.ply --source-map <asset>/projected_manifold.npz \
    --scene $DTU/<scan> --images $DTU/<scan>/images --out experiments/observation_evidence/<scan>_photometric.npz
  python scripts/export_asset_bundle.py --gaussians <asset>/projected_gaussians.ply \
    --mesh <asset>/patch_mesh.ply --meta <asset>/patch_mesh_meta.npz --source-map <asset>/projected_manifold.npz \
    --observation-evidence experiments/observation_evidence/<scan>_photometric.npz \
    --max-observation-photometric-std-percentile 90 --out $OUT/<scan>_vanilla_matched/hybrid_asset
  # 再跑冻结 benchmark
  python scripts/run_asset_benchmark.py --bundle $OUT/<scan>_vanilla_matched/hybrid_asset
  # 汇总三场景表
  python scripts/summarize_asset_benchmark.py $OUT/scan24.../hybrid_asset $OUT/scan65.../hybrid_asset \
    $OUT/scan105.../hybrid_asset --markdown experiments/asset_table.md --csv experiments/asset_table.csv
  ```
- **结果**：edit(P0.3) 与 texture(P0.5) 三场景全 **PASS**。

  | 场景 | certified 泄漏 | baseline 泄漏 | 泄漏减 | 往返 PSNR | overall |
  |---|---:|---:|---:|---:|:--:|
  | scan24 | 0 | 0.1474 | 0.1474 | 30.11 dB | PASS |
  | scan65 | 0 | 0.2814 | 0.2814 | 35.34 dB | PASS |
  | scan105 | 0 | 0.1351 | 0.1351 | 33.70 dB | PASS |

- **结论**：certified 编辑绑定零泄漏由构造保证；信息量在 baseline 确实泄漏（0.135–0.281）。
  texture 往返 30–35 dB 达标；真实杠杆是颜色源质量而非 UV atlas（见 RESULTS §4.1/4.3）。
- **结果文件**：
  - bundle：`$OUT/<scan>_vanilla_matched/hybrid_asset/`
  - 每场景 benchmark：`$OUT/<scan>_vanilla_matched/hybrid_asset/asset_eval/asset_benchmark.json`
  - 汇总表：`experiments/asset_table.md` / `experiments/asset_table.csv`
  - evidence cache：`experiments/observation_evidence/<scan>_photometric.npz`
- **提交**：`4150ec3`；叙事 `RESULTS-LATEST.md` §4.4。

---

## E2 · P0.4 collision GT 对齐（确定性，无需 ICP）

- **目的**：把 DTU 官方 stl（mm，DTU 世界帧）对齐到 Gaussian/重建帧，做 collision-vs-GT。
- **方法**：预处理 `cameras.npz` 的 `scale_mat`（均匀缩放+平移相似变换）即两帧变换；
  `X_norm = (X_mm homogeneous) @ scale_mat_inv_0.T`。均匀缩放保法线方向，stl 自带法线复用。
- **验证**：重建 mesh → 变换后 stl 最近邻中位残差 scan105 **0.04% bbox**、scan24 0.085%、
  scan65 0.067% —— 精确，无需 ICP。
- **结论**：文档担心的「对齐口径风险」**彻底消除**，是确定性变换。SuGaR culled mesh 也在
  mm 帧、复用同一 `scale_mat_inv`（见 E6）。
- **结果文件**：GT npz `$OUT/<scan>_vanilla_matched/hybrid_asset/asset_eval/gt_surface_stlNNN.npz`
  （xyz+normals，Gaussian 帧，stl 下采样 ~600k）。
- **提交**：`4150ec3`；叙事 `RESULTS-LATEST.md` §4.4；对齐方法备忘见记忆
  `dtu-stl-gaussian-alignment`。

---

## E3 · 三场景 collision precision（与裁剪无关方向）

- **目的**：用 collision-vs-GT 量候选面几何忠实度（precision 侧，不受背景污染的方向）。
- **命令**：`python scripts/evaluate_collision_candidate.py --candidate <bundle>/collision_candidate.ply
  --gt <bundle>/asset_eval/gt_surface_stlNNN.npz --out <bundle>/asset_eval/collision_fullstl.json`
- **结果**：

  | 场景 | floater%(>1%bbox) | 候选→GT 中位 | 候选→GT p95 | 法线中位° |
  |---|---:|---:|---:|---:|
  | scan24 | 18.25 | 0.121% | 5.13% | 49.5 |
  | scan65 | 0.87 | 0.126% | 0.657% | 51.7 |
  | scan105 | 1.54 | 0.058% | 0.501% | 52.2 |

- **结论**：scan65/105 候选几何干净（floater <2%）；**scan24 有真实 floater 簇**（见 E4）。
  `coverage`/`hausdorff` 被 DTU 背景底盘污染 → 归待办 A（ObsMask 裁剪），benchmark collision
  线保持 `skip`。
- **结果文件**：`$OUT/<scan>_vanilla_matched/hybrid_asset/asset_eval/collision_fullstl.json`
- **提交**：`4150ec3`；叙事 §4.4。

---

## E4 · scan24 floater 根因 + P0.1 可分性诊断

- **目的**：定位 scan24 floater 簇是哪些 patch，并测现有 CPU 观测证据能否分离 GT-floater。
- **方法**：按 patch 归组 collision 面算 floater 率 → 定位；再把 collision patch 标 GT-floater/
  GT-clean，回看其未被 gate 的 per-patch 证据字段。
- **结果**：
  - 元凶 = patch 26/188/143/130/27/370… 每片 100% 面悬浮（离 GT 中位 2–6% bbox），直径
    1.5–2.5× 中位——**低于 3× scale gate、且通过 observation gate**，三道 GT-free 闸全漏。
  - 可分性（scan24，floater vs clean 中位）：`first_hit_view_count` **11 vs 34（−2.42σ，最强）**；
    `photometric_std` **0.073 vs 0.115（−0.77σ，反向！floater 更平滑）**；`max_parallax` −1.36σ。
  - 但**无单一 CPU 门限可无损清除**（`min_first_hit_views` 扫描：scan24 去 63% floater 面积即
    误伤 clean；scan105 的 floater 更难分）。
- **结论**：GT-free 的 sparse+photometric 观测闸对「被相机看到但几何脱离真实表面」的 floater
  只有**部分**区分力 → **实证需要第二版 restricted-rendering Fisher/Jacobian 证书（GPU/A4）**。
  未改任何冻结阈值。
- **结果文件**：分析脚本一次性计算（未落盘中间文件）；输入 = E1 的 manifest `patch_evidence`
  + E3 的 collision 结果 + GT npz。
- **提交**：`a88b7cf`；叙事 §4.5；缺口 `PROJECT-GAPS-ZH.md` P0.1（2026-07-13 增量）。

---

## E5 · P1.3 三轴主表骨架（CPU 轴）

- **目的**：把散落结果收敛成论文 P1.3 主表形态（precision/coverage/appearance/asset-utility 同表）。
- **结果**（三场景）：识别率 42–58% patches（surface area 50–62%）；precision 轴把 scan24
  floater 簇（18.3% unsupported area）与干净的 65/105（<2%）摊开；asset-util 见 E1。
  coverage(recall) 标 pending A、appearance 标 pending GPU。
- **结论**：认证是保守的，precision 与识别率同表暴露质量差异——满足 P1.3「不能靠拒绝刷 Chamfer」。
- **结果文件**：表在 `RESULTS-LATEST.md` §4.6（数据由 E1/E3 结果文件推得）。
- **提交**：`7ce867a`。

---

## E6 · P1.2 外部 asset baseline：collision precision 三方对比

- **目的**：补真外部 baseline（此前 edit/texture 的 baseline 是内部代理，P1.2 不认）。
- **对手**：
  - **Poisson-from-3DGS**：同源定向点 `projected_points.ply`，open3d depth=9 + 密度分位 0.1
    + 输入 bbox 裁剪（膨胀 5%）。
  - **SuGaR native**：已发表 surface-GS 方法，用其 DTU 评测 `culled_mesh.ply`（DTU mm 帧，
    经 `scale_mat_inv` 转回 Gaussian 帧，复用 E2 工具）。
- **命令**：`python scripts/compare_collision_baselines.py`（见该脚本）。
- **结果**（floater% = 假碰撞面比例，越低越好）：

  | 场景 | ours | sugar-culled | poisson |
  |---|---:|---:|---:|
  | scan24 | **18.3%** | 78.7% | 98.5% |
  | scan65 | **0.9%** | 17.4% | 97.4% |
  | scan105 | **1.5%** | 7.7% | 54.1% |
  | coverage@1% ours/sugar/poisson | 26–41% | 46–70% | 49–76% |

- **结论**：ours 全三场景**假面最少**，连最脏 scan24 也远胜 SuGaR/Poisson；SuGaR 居中（有
  watertight regularization）。代价是 coverage 最保守——**用 coverage 换诚实 precision，
  而非靠拒绝刷 Chamfer**，且对手含一个已发表方法。precision 主指标用稳健面积比 floater%
  （Poisson 距离 p95 被点云离群点撑爆）。
- **结果文件**：
  - 对照表/数：`$OUT/collision_precision_comparison.json` 与 `.md`
  - Poisson mesh：`$OUT/<scan>_vanilla_matched/hybrid_asset/baselines/poisson_fair.ply`
  - SuGaR mesh（外部）：`$SUG/<scan>/dtu_native_mesh/culled_mesh.ply`
- **提交**：`642c1d5`（Poisson）、`faa8b32`（SuGaR）；叙事 §4.7；缺口 P1.2。

---

## E7 · P1.2 edit 轴外部对比：结构化可编辑性

- **目的**：给 edit 轴一个**非循环**的外部对比（现有 edit 指标 `edit_region` 由 patch 定义，
  certified 对它天然零泄漏，不能直接比外部方法）。
- **方法**：数各方法连通分量（编辑单元的天然边界）。ours 用认证 patch 数；SuGaR/Poisson 用
  mesh 连通分量。
- **结果**：ours 381–402 patch（单 patch 中位 0.1% 面积）；SuGaR/Poisson **单一连通体占
  97.5–99.5%**；watertight mesh 上唯一的子区域编辑手段 = proximity 切割，泄漏 13.5–28.1%
  （E1 baseline）。
- **结论**：watertight 抽取无结构化编辑边界，被迫用会泄漏的 proximity；认证 patch 提供
  381–402 个认证独立编辑单元，零泄漏。与 E6（collision）共同构成 asset-utility 两条外部证据。
- **结果文件**：由 `culled_mesh.ply` / `baselines/poisson_fair.ply` / `patch_mesh_meta.npz`
  连通性/patch 统计推得（分析脚本一次性计算）。
- **提交**：见下方最新提交；叙事 `RESULTS-LATEST.md` §4.8。

---

## E8 · texture 轴：per-patch charting 必要性消融

- **目的**：texture 外部对比需给 baseline mesh 强加 charting（有争议，同 edit 几何-GT），
  改做可辩护消融——per-patch vs single-chart（整物体一张切平面）。
- **方法**：per-patch = 各 patch 烘 res32 切平面纹理取 PSNR 均值；single-chart = 全部
  attached 点当一个 chart，测 res32 与总纹素匹配（res≈32·√#patches）两档。
- **结果**：per-patch 30–35 dB；single-chart@32 仅 11–16 dB；single-chart@纹素匹配 18–25 dB
  ——**同总纹素预算下 per-patch 仍领先 6–15 dB**。
- **结论**：单切平面无法表示曲面/闭合面（对侧点塌到同一 UV），拓扑限制非分辨率 →
  **per-patch charting 对纹理保真必要**。与 §4.1 互补：charting 决定保真、颜色源决定 seam。
- **结果文件**：由 `attached_gaussians.ply` + `asset_mapping.npz[attached_patch_ids]` 推得
  （分析脚本一次性计算）。
- **提交**：见下方最新提交；叙事 `RESULTS-LATEST.md` §4.9。

---

## E9 · asset 可用性演示：物理 phantom-collision

- **目的**：证明输出**能当 asset 用**（指标之外的端到端演示）——把 collision mesh 当刚体
  碰撞体，看物体会不会撞进虚空。
- **命令**：`python scripts/demo_collision_physics.py --n-probes 120000`
- **方法**：自由空间撒探针（远离 GT 表面），量每个 mesh 有多少探针落在其接触容差内
  （= 幻影碰撞）；并出横截面图。CPU（numpy+scipy+matplotlib Agg）。
- **结果**（phantom-collision rate，越低越好）：ours scan24/65/105 = 0.10/0.00/0.00%；
  SuGaR = 3.58/0.36/0.05%；Poisson = 6.50/5.97/2.90%。
- **结论**：ours 几乎不挡空处 → 物理体只与真实物体接触；Poisson 2.9–6.5% 幻影碰撞；SuGaR 居中。
  横截面图直观展示 Poisson 糊满空处。注意 phantom-collision（体积）≠ floater%（面积）：scan24
  floater 18% 但 phantom 仅 0.1%。这是「collision precision → 物理正确」的端到端实用价值证据。
- **结果文件**：`$OUT/physics_demo/phantom_collision.json`、`<scan>_cross_section.png`；
  入库副本 `figures/physics_demo/`。
- **提交**：见下方最新提交；叙事 `RESULTS-LATEST.md` §4.10。

---

## E10 · asset 可用性演示：语义部件编辑

- **目的**：第二个可用性演示（能被美术干净编辑）。
- **命令**：`python scripts/demo_edit_propagation.py --seed-xyz 0.13 -0.17 -0.26
  --select-radius 0.12 --prox-radius-frac 0.06 --rotate-deg 35`
- **方法**：选密集区一簇 30 个认证 patch（2287 顶点）刚性旋转 35°；比 certified patch
  binding vs proximity binding 的边界泄漏；出 before/after 图。CPU（复用 `edit_metrics.py`）。
- **结果**：certified 泄漏 **0.00%**（0 顶点）；proximity（6% bbox 选区）泄漏 **19.72%
  = 3847 顶点**（比部件本身还多）。
- **结论**：certified 精确移动语义部件、邻域零位移；proximity（watertight baseline 唯一
  可用手段，见 E7）误拖近 20% mesh。零泄漏 edit 从数字落到可视编辑行为。与 E9 共同构成
  asset 可用性的两个 CPU 端到端演示。
- **结果文件**：`$OUT/edit_demo/edit_propagation.json`、`edit_before_after.png`；入库副本
  `figures/edit_demo/`。
- **提交**：见下方最新提交；叙事 `RESULTS-LATEST.md` §4.11。

---

## E11 · asset 可用性演示：引擎级 GLB 封装

- **目的**：产出可直接拖进 Blender/Unity/Godot 的标准资产文件（可用性最硬交付物）。
- **命令**：`python scripts/package_asset_glb.py`
- **方法**：把 bundle 封装为 `.glb` 场景，含两节点——`certified_patches`（认证网格按 patch
  确定性调色板着色）+ `collision_candidate`（半透明碰撞代理）。trimesh 导出 + round-trip 校验。
- **结果**：三场景 glb 各 580 KB–1 MB，2 节点，faces 26k–47k，回读校验通过。
- **结果文件**：`$OUT/<scan>_vanilla_matched/hybrid_asset/asset_glb/<scan>_hybrid_asset.glb`；
  入库副本 `assets/glb/`。
- **提交**：见下方最新提交；论文 §7.8。

---

## 待办 / GPU 侧（未做）

- **待办 A（已拍板暂缓）**：collision `coverage`/`hausdorff` 需 DTU 官方 ObsMask+Plane 裁剪
  才可比可入 gate（`.../ObsMask/ObsMaskNNN_10.mat` + `PlaneNNN.mat` 已在盘）。未裁前 benchmark
  collision 线保持 skip。
- **edit/texture 外部对比**：把 Poisson/SuGaR mesh 绑回 Gaussian 做编辑传播与纹理往返（进行中）。
- **2DGS mesh** 走同一 collision/edit/texture 口径（GPU）。
- **appearance 轴**：held-out PSNR/SSIM/LPIPS（GPU 渲染）。
- **P0.1 第二版**：restricted-rendering Fisher/Jacobian 几何可识别性证书（GPU/A4）——E4 已实证其必要性。
