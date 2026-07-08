# Manifold-Conservative Gaussian Splatting：从守恒离散测度到可认证的混合几何资产

> 中文论文初稿 v0.1，2026-07-03。本文以 `CLAIM-EVIDENCE-ZH.md` 为唯一事实口径；
> 结果均保留协议限定，不将 pilot、post-hoc diagnostic 或负结果改写为普适结论。

**作者：**【待补】  
**单位：**【待补】  
**目标会议：**【待定】

## 摘要

3D Gaussian Splatting（3DGS）以可微光栅化和高质量新视角合成著称，但其自适应
densification、clone、split 与 prune 主要服务于外观优化，并未规定几何量在表示细化
过程中应如何传递。因此，即使单个 Gaussian 已被约束为薄盘并带有法向或曲率，整个
splat 集合仍可能随 refinement 改变其几何质量，或形成局部自洽但与真实场景不一致的
曲面。本文提出 **Manifold-Conservative Gaussian Splatting**，将自适应 Gaussian
集合表述为 refinement-conservative 的离散几何测度。具体而言，我们从 covariance
谱提取局部维度与 tangent plane，为每个 primitive 引入独立于 appearance opacity 的
几何求积质量 $q_i$，并构造在 clone/split 中严格守恒质量、重心和适用切向矩的传输
规则。由此得到离散 varifold

$$
V_G=\sum_i q_i\,\delta_{(\mu_i,P_i)},
$$

其对 bounded-Lipschitz 测试函数的 refinement 扰动可由子节点空间位移与切平面误差
显式控制。为判断独立学习的 support、tangent 与 curvature 是否可能来自同一局部
曲面，我们进一步在置信度筛选的 MLS chart 上构造基本形式 compatibility residual，
并仅对通过 conditioning、alignment 与 cache-drift 阈值的区域进行投影和 open-patch
提取。

在输出端，我们不将全部 Gaussian 强制闭合为单一 mesh，而是把 certified surface
subset 投影为带边界的 patch mesh，同时保留与 patch 绑定的 appearance Gaussians 和
未认证 residual layer，形成面向编辑、碰撞代理和后续 texture baking 的混合 asset
backbone。当前三 seed sphere 结果中，patch-mesh Chamfer 均值由 0.28287 降至
0.16346（42.0%），certified mass coverage 从 54.95% 提升至 63.44%；但 paired-CI
判定仍为 INCONCLUSIVE，因此这些结果只作为 asset 方向的机制证据。

本文同时给出一个关键边界：**realizability 不等于 identifiability**。Bonnet-type
compatibility 只能约束一个几何场是否可由某个曲面实现，不能仅凭 sparse RGB 选择
ground-truth 曲面。解析场景中的 depth/noise/bias ladder 与注册的 RGB-only 失败实验
共同验证了这一分解。在真实 DTU 场景上，不含 data anchor 的 matched 三场实验未通过
冻结判据（平均 overall 仅改善 0.271%）；加入固定 RGB-SfM 稀疏 support 后，在排除
discovery 场景的两个前瞻复验场景中，官方 overall metric 相对 matched 3DGS 平均改善
2.97%，held-out PSNR 未下降。结果说明，守恒与可实现性约束可作为几何表示原则，但
稳定重建仍需要具有 coercivity 的观测证据。

**关键词：** Gaussian Splatting；varifold；离散几何测度；refinement conservation；
局部可实现性；identifiability；稀疏视角重建

## 1. 引言

3DGS 将场景表示为带位置、covariance、opacity 与 appearance 的 Gaussian primitive
集合，并在训练中通过 densification 动态改变 primitive 数量。该机制对渲染十分有效，
却留下一个基础问题：当一个 Gaussian 被 clone、split、merge 或 prune 时，它所代表的
几何面积和切向信息应如何变化？标准 3DGS 中的 opacity 是外观合成参数，会被遮挡、
颜色误差与 alpha blending 共同驱动，因而不能直接作为几何面积。若将 opacity 同时
解释为 surface mass，则一次外观驱动的 densification 就可能无意中创造或删除几何。

现有 geometry-aware GS 已从多个方向改善这一问题。2DGS 直接采用局部 disk primitive；
SuGaR、GOF、MILo 等方法强化 surface alignment 或 mesh extraction；SolidGS、PGSR、
GausSurf 等引入 depth、normal、MVS 或更稳定的渲染几何；FeatureGS、ARGS 与 GeoSplat
则使用 covariance 谱、rank、curvature 或高阶局部先验。它们已充分说明“让 Gaussian
更薄”或“添加 normal loss”本身不是新的研究问题。本文关注不同层次的缺口：

1. 局部 surface-likeness 并不自动定义对 refinement 稳定的集合级几何测度；
2. tangent、support 和 curvature 分别合理，并不保证它们来自同一局部曲面；
3. 几何场可以被某个曲面实现，也不意味着 sparse observations 能唯一识别该曲面；
4. 未观测区域不应因 mesh extraction 的需要而被伪造成 watertight topology。

据此，本文不宣称“首次将 manifold/varifold 用于 GS”，也不声称解决了通用 sparse-view
surface reconstruction。我们的贡献是以下三个相互衔接且可检验的部分：

- **守恒表示。** 将 Gaussian 集合映射为带独立几何质量的离散 varifold，并给出
  clone、split、merge 与 prune 的守恒或有界扰动规则。对于来自光滑曲面的规则采样，
  该表示具有直接的 bounded-Lipschitz 前向一致性界。
- **可实现性认证。** 在可靠局部 chart 上联合检查 tangent、shape operator、第二
  基本形式 symmetry 与 Gauss compatibility，并将 conditioning、邻域稳定性和 cache
  drift 纳入置信度。输出只覆盖 certified 区域，采用 open patches 而非强制闭合 mesh。
- **可认证的混合 asset 输出。** 将可靠 surface subset 导出为 open patch mesh，保存
  patch ID 与 source Gaussian 对应，并把不可靠或非曲面的 primitive 留在 residual
  splat layer，而不是用全局 Poisson 补出未观测曲面。
- **可辨识性分解。** 从理论与实验上区分 compatibility 控制的 realizability 误差和
  data evidence 控制的 ground-truth 误差。我们报告 RGB-only 的失败，也报告加入固定
  RGB-SfM anchor 后在两个 DTU replication 场景上的条件性改善。

## 2. 相关工作

### 2.1 Gaussian Splatting 中的曲面几何

3DGS 以各向异性三维 Gaussian 表达辐射场。2DGS 将 primitive 改为有向二维 disk，
通过 depth distortion 与 normal consistency 改善几何。SuGaR 将 Gaussian 对齐到
隐式表面并提取、细化 mesh；GOF 从 opacity field 自适应提取 level set；MILo 在训练
中引入 mesh 与 Gaussian 的双向约束。PGSR、GausSurf、SolidGS 和 MeshSplat 分别从
多视几何、深度/法向先验、solid kernel 或 generalizable prior 改善 surface
reconstruction。本文不与这些方法争夺“surface-aware GS”这一宽泛定位，而是研究
primitive 数量变化时离散几何测度的守恒，以及局部几何场能否被同一曲面实现。

### 2.2 Covariance 几何与高阶先验

Gaussian covariance 的 eigensystem 已广泛用于 planarity、effective rank、normal 和
curvature 估计。FeatureGS、ARGS 和 GeoSplat 等工作表明谱与高阶几何先验可有效改善
重建。本文的区别不在于再次使用 eigenvalue loss，而在于把 covariance-induced tangent
与显式质量组合成集合级测度，并规定该测度在 refinement 中的传输规律。尤其是，
appearance opacity 与 geometric quadrature mass 被明确分离。

### 2.3 Varifold、MLS 与基本形式 compatibility

Varifold 是定义在“位置 $\times$ 无向切平面”空间上的 Radon measure，适合表示无需
全局定向、允许 multiplicity 且可比较不同采样的广义曲面。其 kernel distance 已用于
无对应形状匹配与点云学习；离散 varifold 的 first variation 也用于 curvature 估计。
本文不重新发明这些工具，而将其用于分析自适应 Gaussian refinement。局部 MLS 提供
support chart；Gauss–Codazzi 与 Bonnet 理论描述第一、第二基本形式来自同一曲面的
compatibility 条件。我们只将其作为 realizability 诊断，而不把经典存在定理误写成
ground-truth recovery theorem。

## 3. Refinement-Conservative Gaussian Measure

### 3.1 从 radiance Gaussian 到离散 varifold

设 Gaussian 集合为

$$
\mathcal G=\{(\mu_i,\Sigma_i,\alpha_i,c_i,q_i)\}_{i=1}^{N},
$$

其中 $\mu_i\in\mathbb R^3$，$\Sigma_i=R_i\operatorname{diag}
(\lambda_{i1},\lambda_{i2},\lambda_{i3})R_i^\top$，且
$\lambda_{i1}\geq\lambda_{i2}\geq\lambda_{i3}>0$。$\alpha_i$ 与 $c_i$
分别服务于 appearance compositing；$q_i\geq0$ 是独立的 geometric mass。对于
surface-like primitive，最小特征向量给出无符号法向，前两个特征向量张成 tangent
plane，其 projector 为

$$
P_i=R_i[:,1\!:\!2]R_i[:,1\!:\!2]^\top=I-n_in_i^\top.
$$

离散几何对象定义为

$$
V_G(\varphi)=\sum_{i=1}^N q_i\varphi(\mu_i,P_i).
$$

由于 $P_i$ 对 $n_i$ 的符号不敏感，该表示不要求全局一致定向。实现中 covariance
spectrum 还用于给 primitive 标注 surface/curve/volume 候选类型，但类型阈值只是
几何诊断，不等价于 semantic ground truth。

### 3.2 为什么质量必须独立于 opacity

Opacity 决定沿光线的 alpha compositing；它会因可见性、颜色解释和重叠 Gaussian
而变化。几何面积则应近似曲面求积权重。本文因此保存独立 $q_i$，并在缺少显式质量时
仅以 winsorized kNN 面积估计初始化。后者在局部均匀二维采样下近似合理，但在边界、
非均匀密度、噪声或 duplicate layers 下存在偏差，不能被当作无条件面积定理。

### 3.3 前向一致性

设 $M$ 为紧致 $C^2$ 嵌入曲面，面积测度为 $A$，切平面 projector 为 $P(x)$。
对分区 $\{C_i\}$，若

$$
\sup_{x\in C_i}\|x-\mu_i\|\le h,\qquad
\sup_{x\in C_i}\|P(x)-P_i\|_F\le\varepsilon_T,
$$

且 $\sum_i|q_i-A(C_i)|\le\varepsilon_q$，则对 supremum norm 与 Lipschitz constant
均不超过 1 的任意测试函数 $\varphi$，有

$$
|V_G(\varphi)-V_M(\varphi)|
\le A(M)(h+\varepsilon_T)+\varepsilon_q.
$$

证明只需插入 cell-area quadrature 并使用三角不等式。该命题说明 covariance-to-measure
映射在采样、切向和质量误差受控时是一致的；困难在于训练是否产生这些假设，而不在
于把有限 atomic measure 与光滑曲面直接等同。

### 3.4 守恒 refinement

将 parent atom $(q,\mu,P)$ 替换为 children $(q_j,\mu_j,P_j)$。若

$$
\sum_jq_j=q,\qquad \sum_jq_j\mu_j=q\mu,
$$

则零阶质量与一阶空间矩严格守恒；若进一步有
$\sum_jq_jP_j=qP$，切向 projector moment 也守恒。对任意 1-Lipschitz
$\varphi$，refinement 扰动满足

$$
\left|\sum_jq_j\varphi(\mu_j,P_j)-q\varphi(\mu,P)\right|
\le\sum_jq_j\bigl(\|\mu_j-\mu\|+\|P_j-P\|_F\bigr).
$$

我们的 conservative split 令 children 继承 parent tangent，并对偏移重新中心化，
因而三项等式确定性成立。Merge 一般保持质量和重心；只有在 child tangent 一致时
严格保持切向矩。Prune 并非守恒 split/merge：实现将删除质量运输到最近保留点，严格
保持总质量，但其一阶矩变化仅有

$$
\left\|\Delta\sum_iq_i\mu_i\right\|
\le\sum_{i\in\mathcal R}q_i\|\mu_i-\mu_{a(i)}\|.
$$

这一边界通过 prune ledger 显式记录。完全保持局部一阶矩的非负 barycentric prune
仍属于后续工作。

## 4. Confidence-Certified Local Realizability

### 4.1 局部 chart 与 support

对候选 surface primitive 建立 kNN 邻域，在局部 frame $(e_1,e_2,n)$ 中以加权最小
二乘拟合 Monge chart

$$
X(u,v)=(u,v,f(u,v)).
$$

由 chart 得到 support normal、metric $g$、第二基本形式 $b$ 与 shape operator
$S=g^{-1}b$。Covariance 同时给出独立预测的 normal/tangent。两条几何来源使我们能够
检测“局部点的位置关系”和“Gaussian 自身声称的切向”是否兼容，而不只是平滑法向。

### 4.2 Compatibility residual

实现使用局部法向导数构造预测第二基本形式 $b_{\mathrm{pred}}$，并计算：

- tangent/support alignment residual；
- $b_{\mathrm{pred}}$ 的 symmetry residual；
- predicted shape operator 与 support shape operator 的差；
- predicted Gaussian curvature 与 support Gaussian curvature 的差。

其中 shape residual 是主要 coercive 项；单独的 symmetry、Gauss 或 Codazzi residual
存在较大零空间，不能控制 support 与 tangent 的实际偏差。在线性化图模型中，若
$r_0$ 表示法向/一阶兼容误差，$r_1$ 表示 shape mismatch，则到 compatible graph
集合 $Z$ 的距离满足条件界

$$
\operatorname{dist}((f,p),Z)
\le C\bigl(\|r_0\|_{L^2}+\|r_1\|_{L^2}\bigr).
$$

在斜率、曲率和法向夹角有界时，该结论可扩展到 nonlinear chart，并额外引入 MLS、
采样和质量误差。若 chart 近乎竖直、Gram matrix 病态、covariance normal 与 support
normal 近乎正交，或邻域混合多个 sheet，常数会退化；这些区域必须拒绝，而不能简单
赋予更大的 loss weight。

### 4.3 认证与 open-patch 输出

认证同时检查 covariance spectrum、局部 support、normal alignment、Gram conditioning、
kNN margin 与 cache drift。只有通过阈值的 primitive 才进入几何投影与 patch mesh。
相邻 certified samples 依据距离、切向和 sheet consistency 连接。输出是可为空、可有
边界、可多连通的 open patches；方法不在未观测区域补洞，也不声称恢复全局 topology。

## 5. Realizability 与 Identifiability

令 $\Pi_ZV$ 表示将离散场投影到 compatible surface-varifold 类 $Z$。对任意 varifold
metric，三角不等式给出

$$
d_V(V,V_*)\le d_V(V,\Pi_ZV)+d_V(\Pi_ZV,V_*).
$$

第一项是 realizability error，可由 compatibility、采样和求积误差控制；第二项比较
两个已经可实现的曲面，必须依赖观测证据。一个完全平滑且满足 Gauss–Codazzi 的错误
平面可令第一项接近零，却仍与 ground truth 相距很远。

对 implemented Gaussian kernel varifold，任意 coupling $\pi$ 给出

$$
\operatorname{MMD}_k(\mu,\nu)
\le\left[\int\left(
\frac{\|x-y\|^2}{\sigma^2}+\frac{\|P-Q\|_F^2}{\tau^2}
\right)d\pi\right]^{1/2}.
$$

守恒 transport 因而提供从空间/切向运输误差到 evaluator 的充分界，但反向不成立：
固定 bandwidth 的小 MMD 可能隐藏细尺度几何或 topology 错误。

已知相机下的可靠 visible depth graph 提供建设性的 coercivity：depth 的 $H^1$ 误差
同时控制位置和 tangent projector，进而控制对应可见区域的 varifold 距离。RGB-only
只有在 restricted rendering Jacobian 在 compatible tangent space 上具有严格正的最小
奇异值时才局部可辨识。低纹理、遮挡、少视角和未观测区域都会破坏这一条件。本文的
RGB-SfM anchor 并非 oracle depth，而是由输入 RGB 经 COLMAP 得到的稀疏几何证据；
它改善 coercivity，但不消除其噪声与 coverage 限制。

## 6. 优化、认证与 Asset Generation

训练在 3DGS photometric objective 上增加几何项。Warm-up 后，根据 covariance
spectrum 建候选 graph，并周期性更新 MLS compatibility cache。损失包括 thinness、
rank-2 neighborhood、normal consistency、area/curvature-scale，以及经 geometric mass
与 confidence 加权的 shape、symmetry 和 Gauss residual。Compatibility 采用 ramp-in；
当 normalized cache drift 超阈值时重建 cache。

训练中的 clone/split/prune 同步更新 $q_i$。固定 RGB-SfM anchor 版本额外将 Gaussian
support 约束到输入 COLMAP 稀疏点的 trust region。离线阶段重新计算 diagnostics，
只对 certified subset 做 MLS projection 和 patch extraction。该设计将“可渲染 radiance
layer”和“可信 geometric asset layer”分开：前者可保留 volume-like residual，后者
宁可拒绝也不制造几何。

### 6.1 混合资产结构

下游输出定义为

$$
\mathcal A=(\mathcal M_{\mathrm{patch}},\mathcal G_{\mathrm{attached}},
\mathcal G_{\mathrm{residual}},\mathcal C),
$$

其中 $\mathcal M_{\mathrm{patch}}$ 是 certified open-patch mesh；
$\mathcal G_{\mathrm{attached}}$ 保存可追溯到 mesh vertex/patch 的完整 appearance
Gaussian；$\mathcal G_{\mathrm{residual}}$ 保存 curve-like、volume-like 或低置信度
primitive；$\mathcal C$ 是 certification metadata，包括 source index、patch ID、质量、
confidence 和拒绝原因。该结构避免把毛发、薄杆、半透明区域或未观测区域错误三角化。

当前 closed path 已支持从 3DGS PLY 生成 projected full-attribute Gaussian PLY、patch
mesh PLY 及 source-Gaussian/patch 对应的 NPZ metadata，并可重新载入官方 3DGS 继续训练。
我们进一步实现标准化 hybrid bundle：按 patch 分组的 OBJ、对应 PLY、完整属性的
attached/residual Gaussian PLY、collision candidate、source-ID mapping 与 JSON manifest。
它已经是 asset backbone，但还不是 production-ready asset package：GLB、UV atlas、
texture baking、经过物理验证的 collision simplification 和编辑传播仍需接入。

### 6.2 面向下游任务的接口

该混合表示针对四类应用设计：

1. **局部几何编辑：** 用户选择一个 certified patch，顶点变形通过 source mapping
   传播到 attached Gaussians；residual layer 不被错误拖拽。
2. **碰撞与导航代理：** 只将高置信 patch mesh 简化为 collision mesh，并把未认证区
   标为 unknown，而不是 free space。
3. **Texture/material baking：** 在每个 chart 内展开 UV，将多视颜色或 Gaussian
   appearance 烘焙到局部 texture；patch boundary 保留为可见的不确定性边界。
4. **混合渲染与 LOD：** 近景使用 attached splats，远景使用简化 mesh；curve/volume
   residual 继续由 splat renderer 表达。

论文的完整下游验证不能只展示 mesh 截图。至少应测量 edit propagation error、
collision false-positive/false-negative、texture reprojection quality、mesh simplification
曲线，以及 asset round-trip 后的 rendering degradation。这些度量的 CPU 版本已实现
（编辑传播、collision coverage/false-surface-vs-tolerance、texture round-trip 与 seam），
并在 sphere 与真实 scan105 backbone 上给出首批数字（见 §7.5）；仍缺的是经渲染器的
round-trip PSNR/SSIM 与外部 binding 对照。

## 7. 实验

### 7.1 实验问题与指标

实验回答五个问题：守恒算子是否避免 refinement 引起的测度漂移；compatibility 在何种
data evidence 下有效；certified subset 能否产生更可靠的 asset backbone；RGB-only 的
失败是否符合 identifiability 分析；真实场景上的增量能否在 matched resource schedule
下复现。

我们报告 certified Chamfer、normal angular error、normalized kernel-varifold distance、
certified mass/coverage、mesh Chamfer、DTU accuracy/completeness/overall，以及 held-out
PSNR/SSIM。任何单一指标均不足以代表“总体最好”。

### 7.2 守恒机制验证

单元测试验证 conservative clone/split 的总质量、重心和 inherited-tangent moment 在
数值精度内不变；prune 总质量严格守恒，其运输代价和一阶矩上界由 ledger 记录。
对同一几何做不同 refinement schedule 时，显式 $q_i$ 的 varifold 诊断明显比将 opacity
当面积更稳定。【定稿需从测试汇总自动填入误差均值、最大值与 schedule 数量。】

### 7.3 解析 identifiability ladder

在由同一解析 geometry source 生成 RGB、depth、normal、mask、surface sample 与 mesh
GT 的 plane/torus 场景上，我们依次测试 exact depth、noisy depth、affine-biased depth、
calibrated depth 与 RGB-only。可靠或经校准的 depth anchor 与 compatibility 联合时，
位置与 tangent error 显著下降；结构化 bias 可收敛到残差很低但位置错误的可实现曲面。
注册的 RGB-only 7k 实验总体判定为 FAIL，fixed-support/trust-region 多 seed 结果为
0 PASS、1 TRADEOFF、3 FAIL。这些负结果与理论一致：compatibility 收缩到 $Z$，但不负责
在 $Z$ 中选择 $V_*$。

### 7.4 外部方法 pilot

在 plane/torus、3 train views 与 12 held-out views 的简单协议中，本方法 7k 相对官方
2DGS 30k 获得更低的 Chamfer、normal error 和 kernel-varifold distance；但 plane 上
2DGS 三个 run 均 collapse，该结果只说明窄协议下的数值差异，不可外推为普遍领先。

SuGaR 8GB pilot 呈现互补结果。SuGaR 在 Chamfer、normal 与 mesh Chamfer 上明显更好；
本方法在部分 varifold/coverage 指标上占优。由于 SuGaR 仅 seed 0 且采用缩减的 8GB
配置，本实验用于刻画差异而非生成排名。GeoSplat 当前无同协议可运行官方实现，因此
只做概念层面对照。

### 7.5 Asset backbone 机制实验

在 sphere sparse-view 三 seed 实验中，我们比较 vanilla、仅一阶 tangent supervision
和完整 compatibility pipeline。完整方法的 certified mass coverage 从 vanilla 的
54.95% 提升至 63.44%，certified point Chamfer 从 0.24090 降至 0.15211，sampled
patch-mesh Chamfer 从 0.28287 降至 0.16346；后两者按均值分别改善 36.9% 和 42.0%。
所有输出 mesh 的 non-manifold edge 数为 0。代价是 held-out PSNR 从 21.4959 降至
21.3542 dB，SSIM 从 0.5539 降至 0.5395。

该结果表明 certification 与 compatibility 有潜力把 radiance Gaussians 转换为更可靠的
surface backbone，但三 seed paired confidence interval 尚未整体越过预设改善阈值，
正式判定是 **INCONCLUSIVE**。此外，当前 mesh 仍可能包含较多 connected components
和 boundary edges；这些边界是拒绝未知区域的设计结果，不应与 watertight completeness
混为一谈。

当前实验尚未闭合真正的 downstream asset loop。定稿前需增加：同一模型上的 patch
选择与形变传播、collision proxy、UV/texture baking、mesh simplification 和 Blender/
engine round-trip，并与 SuGaR mesh binding、Poisson mesh 及 matched 3DGS extraction
比较。没有这些实验时，本文只能声称输出“asset-ready backbone”，不能声称已生成
production-ready asset。

作为导出完整性检查，sphere seed-0 的 projected checkpoint 含 2,625 个 Gaussians；
bundle 将其中 1,656 个绑定到 28 个 certified patches，其余 969 个保留为 residual。
输出 mesh 含 2,866 个 triangles、464 条开放边界和 0 条 non-manifold edge。用于碰撞
候选时，基于相对 patch 尺度拒绝 1 个异常 component，保留 27 patches、2,836 个
triangles。该数字只
验证分层、映射和拓扑守卫能够工作，不用于证明下游编辑优于基线。

我们进一步实现并运行了三条 asset-utility CPU 度量（编辑传播、collision coverage、texture
round-trip），给出首批可量化数字，并暴露两个方法学要点。在 sphere seed-0 上，certified
patch binding 的编辑传播为零边界泄漏、零 residual 污染，而 nearest-radius baseline 泄漏
12.9%、污染 13.9%，验证 source-mapping 绑定的隔离性。collision candidate 对解析 GT 在
1% bbox tolerance 下 false-surface fraction 达 74%，但 tolerance 扫描表明这主要是评测口径
产物：candidate→GT 中位误差为 0.033，略高于该 tolerance，放宽到 2% bbox 时 coverage 即
达 72%；且 circumradius/alpha bridge 过滤对 false fraction 无改善，说明 false surface 均匀
分布、来自 certified 几何自身精度而非切向外推。texture round-trip 在真实 scan105 bundle 上
显示逐 patch seam 主要由观测颜色源质量决定：噪声 SH-DC 下 seam PSNR 为 12.5 dB，改用
多视 photometric mean color 后升至 18.36 dB（+5.86 dB），但两种颜色源下 baked seam 都与其
raw-color ceiling 基本相同，说明剩余 seam 是跨 patch 边界的真实颜色方差、共享 UV atlas
无法进一步消除。此外，我们在 scan105 上首次把 observation identifiability gate 端到端接入
asset 导出：402 个 patch 中 139 个因 sparse support 不足、29 个因 multi-view photometric
不一致（每场景相对百分位门限）被拒，234 个通过认证。以上说明相应 CPU 度量已就绪，但真实
多场景、经渲染器的 round-trip 以及与 SuGaR/Poisson binding 的对照仍未完成。

### 7.6 DTU matched diagnostic：无 anchor 的失败

我们在 DTU scan24、65、105 上采用相同半分辨率 7k schedule、固定 7-view held-out
split、近似相同 Gaussian 数、相同 patch extractor 与官方 evaluator。相对 matched
3DGS，无 anchor 的 manifold pipeline 在 scan24/65 上 overall 分别改善 0.124% 和
2.54%，在 scan105 上退化 1.85%。三场 mean relative overall 仅改善 0.271%，低于
预先冻结的 1% 门槛，故总体判定为 **FAIL (3/3)**。

三场平均 accuracy 改善 1.98%，completeness 平均退化 0.38%，同时 PSNR 与 Gaussian
数量 guardrail 通过。这表明当前内部 realizability 约束更像已观测表面的 precision
regularizer，并未提供填补 coverage 的 data-coercive 信号。

### 7.7 固定 RGB-SfM anchor 的条件性复验

在看到无 anchor 失败后，我们先在 scan105 做 post-hoc discovery。加入固定 COLMAP
anchor 后，相对 matched 3DGS 的 overall 从 0.9831 降至 0.9195，改善 6.47%，其中
completeness 改善 8.92%，PSNR 下降 0.210 dB。该场景仅用于形成机制假设，不纳入
replication 统计。

随后冻结规则并在 scan24/65 上复验：两场 overall 必须均改善，平均改善至少 1%，
每场 PSNR delta 不低于 $-0.3$ dB，Gaussian 数差不超过 5%。结果如下。

| Scan | Matched 3DGS overall ↓ | 本文（+固定 COLMAP anchor）↓ | 相对改善 | PSNR delta |
|---|---:|---:|---:|---:|
| 24 | 1.7163 | **1.6885** | **1.62%** | +0.132 dB |
| 65 | 2.4643 | **2.3577** | **4.32%** | +0.297 dB |
| **均值** | — | — | **2.97%** | +0.215 dB |

两场均通过，最大 Gaussian 数差为 0.38%，因此 replication 判定为 **PASS (2/2)**。
该结果支持的严格结论是：在固定 RGB-SfM 稀疏 support、matched resource schedule 和
两个前瞻 DTU 场景下，realizability pipeline 可复现地改善官方 overall metric，且不
损失 held-out PSNR。它不恢复 RGB-only claim，也不构成相对 2DGS、SuGaR、GeoSplat
或其他 SOTA 的总体领先结论。

## 8. 讨论

实验揭示了三个层次不可互换。第一，thin covariance 是 primitive-level property；
若 $q_i$ 随 densification 漂移，集合级 geometry 仍不稳定。第二，compatibility 是
field-level property；它排除无法来自同一曲面的 tangent/curvature 组合，却允许许多
错误但可实现的曲面。第三，identifiability 是 observation-level property；只有 depth、
RGB-SfM anchor 或满足局部 Jacobian coercivity 的多视 RGB 才能在 compatible class 中
选择 ground truth。

这也解释了 DTU 结果：无 anchor 的 loss 能提高局部 accuracy，却无法稳定改善
completeness；稀疏 COLMAP support 提供了 coverage 方向的数据约束，使相同的
realizability machinery 在两个 replication 场景上产生稳定增量。2.97% anchor 增益
的作用是验证 data-coercivity 分解，而不是充当论文的主要性能卖点。本文方法更适合被
理解为“带拒绝机制、面向下游资产化的几何表示层”，而不是单独解决重建歧义的万能
regularizer。

## 9. 局限性

- 当前真实实验仅覆盖三个 DTU diagnostic 场景和两个正式 replication 场景，统计规模
  不足以支持 SOTA 或广泛泛化结论。
- RGB-SfM anchor 依赖 COLMAP support 的覆盖、噪声与 calibration；低纹理或重复纹理
  场景仍可能失败。
- 局部 MLS chart 与 compatibility 不保证全局 topology，固定 bandwidth varifold MMD
  也可能忽略细尺度错误。
- Merge 对变化 tangent 不能严格保持 projector moment；prune 仅保持总质量，并以运输
  界控制一阶矩。
- 当前 SuGaR 仅为缩减 pilot，GeoSplat 缺同协议实现；仍需完整预算、多 seed、统一
  extractor 的外部 baseline。
- Curve/volume mixed-dimensional 分支已有表示接口，但尚无足够实验支撑为主贡献。
- 当前 asset 输出缺少 UV/material baking、collision proxy、编辑传播与标准 GLB/OBJ
  封装；“asset-ready backbone”与“可直接用于生产的 asset”之间仍有工程和实验缺口。
- asset-utility 度量（编辑/collision/texture）虽已实现并在 sphere 与 scan105 上运行，
  但首批结果显示 collision 的高 false-surface 与 texture seam 这两个表观弱点主要由评测
  口径决定（tolerance 相对方法精度过紧、观测颜色源为噪声 SH-DC），真实杠杆是几何精度
  与多视颜色质量；真正的 UV atlas、经渲染器的 round-trip PSNR/SSIM 与外部 binding 对照仍缺。

## 10. 结论

本文将自适应 Gaussian splats 表述为 refinement-conservative 的离散几何测度，并把
geometric mass 与 appearance opacity 分离。守恒 transport 使 clone/split/merge/prune
对 varifold 的影响可被精确保持或显式界定；confidence-certified 基本形式
compatibility 则检测 support、tangent 与 curvature 的局部可实现性。更重要的是，本文
证明并实验展示了 realizability 与 identifiability 的边界：前者不能替代观测证据。
真实 DTU 的失败与 anchor replication 共同表明，几何一致性必须与具有 coercivity 的
data anchor 结合；sphere asset 实验则显示该表示在 patch-mesh quality 上有比整体 DTU
重建指标更明显的潜在收益。后续工作的关键不是继续放大 2.97% 的 reconstruction 数字，
而是闭合从 certified patch、attached splat 到编辑、碰撞、烘焙和 engine round-trip 的
asset pipeline，并在真实场景上验证其实际可用性。

## 参考文献（初稿占位）

1. Kerbl et al. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH, 2023.
2. Huang et al. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. SIGGRAPH, 2024.
3. Guédon and Lepetit. SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction. CVPR, 2024.
4. Yu et al. Gaussian Opacity Fields. SIGGRAPH Asia, 2024.
5. Charon and Trouvé. The Varifold Representation of Nonoriented Shapes for Diffeomorphic Registration. SIIMS, 2013.
6. Hsieh and Charon. Metrics, Quantization and Registration in Varifold Spaces. FoCM, 2021.
7. Levin. The Approximation Power of Moving Least-Squares. Mathematics of Computation, 1998.
8. Li et al. GeoSplat: Geometry-Constrained Gaussian Splatting. 2025.
9. 【补齐 PGSR、GausSurf、SolidGS、FeatureGS、ARGS、MILo、MeshSplat 与 DTU benchmark 文献。】
