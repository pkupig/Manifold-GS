# 数学补课与行动清单（Bonnet 兼容性这条线）

这份文档专门给"概念暂时还不会"的情况。每个概念按同一格式讲：

> **一句话** → **直觉** → **公式** → **代码在哪**（`manifold_gs/fundamental_compatibility.py`）→ **我们为什么需要它**

目标：读完你能看懂 `THEORY-BONNET.md` 的每一步，并能跟人讲清楚"我们到底加了什么约束、为什么它不平凡"。

所有这些都属于**经典曲面微分几何**（不是前沿数学），标准教材是 do Carmo
《Differential Geometry of Curves and Surfaces》第 2–4 章。下面按依赖顺序排，
**从上往下学**，每个都建立在前一个上。

---

## 0. 前置：切空间与法向（tangent / normal）

- **一句话**：曲面上每一点都有一个贴着它的平面（切平面）和一根垂直于它的方向（法向）。
- **直觉**：站在地球表面，脚下的地面是切平面，头顶朝天的方向是法向。
- **公式**：曲面用参数化 `phi(u,v) -> R^3` 表示。切空间由两个偏导张成：
  `phi_u = ∂phi/∂u`, `phi_v = ∂phi/∂v`；法向 `n = (phi_u × phi_v)/|phi_u × phi_v|`。
- **代码在哪**：`t1, t2`（切向量）、`n0` / `fitted_normal`（法向）。其中
  `n0` 来自邻域中心协方差的最小特征向量，`t1,t2` 来自另外两个特征向量。
- **为什么需要**：3DGS 的每个 splat 自带一个朝向（旋转 `R` 的第三列就是它的法向）。
  整条线的出发点就是把这个朝向当成"曲面法向"来检验。

---

## 1. 第一基本形 I（first fundamental form / 度量）

- **一句话**：在曲面上量"长度和角度"的工具，本质是切平面里的内积。
- **直觉**：地图上两点画一条线，实际地表上这条线多长？I 就是把参数空间 (u,v) 的
  位移翻译成真实空间长度的"换算表"。它是**内蕴**的——只看曲面自己，不看它怎么弯。
- **公式**：`I = J^T J`，其中 `J = [phi_u, phi_v]` 是参数化的雅可比。
  写成矩阵 `I = [[E, F],[F, G]]`，`E=phi_u·phi_u, F=phi_u·phi_v, G=phi_v·phi_v`。
- **代码在哪**：`metric = parametric_basis.T @ parametric_basis`，
  `metric_inv = inv(metric)`。`parametric_basis` 就是上面的 `J`。
- **为什么需要**：要把"第二基本形"变成 shape operator（见 §4），必须用 `I^{-1}` 去
  归一化。Bonnet 定理里 I 是两块输入数据之一。

---

## 2. 第二基本形 II（second fundamental form）

- **一句话**：描述曲面"往哪个方向、弯多厉害"的工具。是**外蕴**的——看曲面在 3D 里
  怎么鼓起来 / 凹下去。
- **直觉**：在切平面上方搭一个小的"高度场" `h(u,v)`（曲面离切平面有多高），II 就是
  这个高度场的二阶项（Hessian）。平面 II=0；球面处处同样弯，II 正比于单位阵。
- **公式（两种等价定义，正是本项目的核心）**：
  - **从位置**（高度场二阶导）：`II_ab = ⟨∂_a∂_b phi, n⟩`。
  - **从法向**（法向的变化）：`II_ab = −⟨∂_a n, ∂_b phi⟩`。
  - 真实光滑曲面上这两者**恒等**（对 `⟨∂_a phi, n⟩=0` 求导即得）。**3DGS 里位置和
    法向是独立变量，于是两者可以不相等——这就是我们要罚的东西。**
- **代码在哪**：
  - `support_second = hessian / sqrt(1+|grad|^2)`：从**位置** Monge 拟合得到的 II
    （§3 会讲 Monge）。
  - `predicted_second = −normal_derivatives.T @ parametric_basis`：从**法向场**得到的
    II = `−∂n·∂phi`。
- **为什么需要**：`support_second` vs `predicted_second` 的差就是 `THEORY-BONNET.md`
  里 `A_support` vs `A_pred` 比较的源头。**整个新颖性就建立在"同一个 II 被两个独立
  来源算出来、然后对账"。**

---

## 3. Monge 二次曲面 / Monge patch（二次高度场拟合）

- **一句话**：在一点附近，把曲面写成"切平面 + 一个二次高度修正"的局部模型。
- **直觉**：任何光滑曲面在一点附近放大看，都像 `z = (a u² + 2b uv + c v²)/2 +
  (一次项)`。这个抛物面近似就是 Monge patch；它的二次系数直接给出 II。
- **公式**：在局部切坐标 `(u,v)` 下最小二乘拟合
  `height ≈ c0 + c1 u + c2 v + 0.5 c3 u² + c4 uv + 0.5 c5 v²`。
  则梯度 `grad=(c1,c2)`，Hessian `[[c3,c4],[c4,c5]]`，II 由 Hessian 给出。
- **代码在哪**：
  ```
  design2 = column_stack([1, u, v, 0.5 u², u v, 0.5 v²])
  coeff   = weighted_lstsq(design2, height, weights)
  gradient = coeff[1:3];  hessian = [[c3,c4],[c4,c5]]
  ```
  （在 `manifold_projection.py::_fit_local_surfaces` 里还有一处类似的二次拟合，用来
  做投影时抓曲率、避免反复投影到平面导致球面收缩。）
- **为什么需要**：它是**只用位置**估计 II 的具体算法（得到 `support_second`）。
  "二次"很关键——一次（线性 PCA 平面）只能给切平面、给不出曲率，没法和法向导出的 II
  对账。

---

## 4. Weingarten 映射 / shape operator（形状算子 S）

- **一句话**：把"沿切方向走一步、法向转了多少"打包成一个 2×2 矩阵。它的特征值就是
  主曲率。
- **直觉**：你在曲面上沿某方向走，法向跟着扭。扭得快=曲率大。S 就是"法向对位置的
  导数"在切平面里的表示。Weingarten 映射 = Gauss 映射（点→法向）的微分。
- **公式**：`S = I^{-1} II`。特征值 `kappa_1, kappa_2` = 主曲率；
  Gauss 曲率 `K = det(S) = kappa_1 kappa_2`；平均曲率 `H = trace(S)/2`。
- **代码在哪**：
  - `support_s   = metric_inv @ support_second`（位置版 S）
  - `predicted_s = metric_inv @ predicted_second`（法向版 S）
  - `predicted_shape`, `support_shape` 是把它们转到统一切标架后的 2×2 矩阵。
- **为什么需要**：直接比 II 受参数化影响；比 shape operator（已被 `I^{-1}` 归一化）
  才是几何上正当的对账。`shape_relative = ‖predicted_shape − support_shape‖/scale`
  就是主指标之一。

---

## 5. 三个曲率（主曲率 / Gauss / 平均）—— 只需会读

- **主曲率** `kappa_1,kappa_2`：两个正交方向上弯得最狠 / 最缓的程度。
- **Gauss 曲率** `K = kappa_1 kappa_2`：**内蕴**（Gauss 绝妙定理：K 只由 I 决定，
  不用看曲面怎么嵌进 3D）。球面 K>0，平面 K=0，马鞍 K<0。
- **平均曲率** `H = (kappa_1+kappa_2)/2`：肥皂膜极小曲面 H=0。
- **代码在哪**：`gauss[i] = |det(predicted_shape) − det(support_shape)| · r²`，
  比较两种来源算出的 K（det of shape operator = K）。
- **为什么需要**：Gauss 方程（§6）就是用 K 把内蕴与外蕴绑起来；它是 Bonnet 的两条
  可积分性方程之一。

---

## 6. Gauss 方程 + Codazzi-Mainardi 方程（兼容性方程）

- **一句话**：I 和 II 不能瞎填——它们必须满足两条相容方程，才可能来自同一张曲面。
- **直觉**：给你一张"度量表 I"和一张"弯曲表 II"，问：世界上存不存在一张曲面同时满足
  这两张表？答案是：**当且仅当**它们满足 Gauss + Codazzi。这是曲面存在性的"守恒律"。
  - **Gauss 方程**：由 II 算的外蕴 K，必须等于由 I 算的内蕴 K。
  - **Codazzi-Mainardi 方程**：II 的协变导数必须对称，`∇_a II_bc − ∇_b II_ac = 0`。
    它本质是说"法向场可积分（是某个嵌入的 Gauss 映射）"。
- **代码在哪**：
  - Gauss：`gauss_residual_scaled`（上面 §5）。
  - Codazzi：第二遍循环里 `codazzi[i]`——把相邻点的 shape operator 搬运到同一标架，
    拟合其一阶变化，取 `∇_a II_bc − ∇_b II_ac` 的范数。
  - `symmetry_residual = ‖predicted_second − predicted_secondᵀ‖`：II 必须对称，
    这是 Codazzi 的一个一阶必要条件（不可积/有 curl 的法向场会破坏对称）。
- **为什么需要**：这是把"法向场是否被某曲面实现"变成**可计算残差**的关键。
  `THEORY-BONNET.md` §3 的反例（平面上的旋转法向场）就是 II 反对称、Codazzi 残差
  非零，而光度 / thinness / 法向平滑全都看不出来。

---

## 7. Bonnet 定理（曲面基本定理）—— 整条线的"为什么有意义"

- **一句话**：满足 Gauss+Codazzi 的 (I, II) **唯一确定**一张曲面（差一个刚体运动）。
- **直觉**：度量 I + 弯曲 II + 两条相容方程 = 曲面的"完整 DNA"。残差归零，等于认证
  "学到的(位置, 法向)对在局部就是一张真曲面，并且把它钉死到差一个平移+旋转"。
- **公式**：见 `THEORY-BONNET.md` §3.3。教材：do Carmo Ch. 4（fundamental theorem
  of the local theory of surfaces）。
- **为什么需要**：它把我们的损失从"启发式正则项"升级成"曲面存在性的充要条件"。这是
  论文叙事的数学骨架——不是"让法向看起来一致"，而是"强制 Bonnet 可实现性"。

---

## 8. varifold（项目里用到，了解即可）

- **一句话**：把曲面看成"位置 × 切平面"上的一个测度，不要求它真是光滑曲面。
- **直觉**：一堆带朝向的点 `sum_i q_i δ_(mu_i, P_i)`，每个点带一个质量 `q_i` 和一个
  切平面 `P_i`。它能描述带洞、带边界、甚至非流形的东西——比"曲面"更宽松。
- **关键点（与本线相关）**：varifold 的朝向 `P_i` **天然可以独立于位置** ——这正好
  对应 3DGS 的位置/法向解耦。但现有 varifold 工作（Buet-Rumpf 等）通常**从点本身**估
  曲率，没有把"独立法向 vs 位置曲率之差"当判别量用（见 `THEORY-BONNET.md` Q4）。
- **代码在哪**：`geometric_measure.py`（质量/矩守恒）、`gt_metrics.py` 的
  kernel-varifold 距离。
- **为什么需要**：这是项目另一条支柱（守恒测度 `q_i`）。Bonnet 这条线可以和它合并成
  一个主张，见 `THEORY-NOVELTY-REAUDIT-2026-06-27.md` §4。

---

## 学习路径（建议顺序与时间）

1. **半天**：§0→§2，看 do Carmo 2-3 节 + Wikipedia "First/Second fundamental form"。
   目标：能手算平面、球面、圆柱的 I 和 II。
2. **半天**：§3→§5，看 do Carmo "Gauss map / shape operator"。
   目标：能解释 `S=I^{-1}II`，主曲率是 S 的特征值。
3. **1 天**：§6→§7，看 do Carmo Ch.4。
   目标：能说出 Gauss+Codazzi 是什么、Bonnet 在保证什么。
4. **0.5 天**：对照 `fundamental_compatibility.py`，把每个变量名标到上面的公式上
   （上面已逐行标好，照着核对一遍即可）。

最省力的两个资源：
- do Carmo, *Differential Geometry of Curves and Surfaces*, Ch.2–4（唯一必读）。
- 3Blue1Brown / 任意"second fundamental form intuition"视频（建直觉，非必须）。

---

## 行动清单（数学之外，推进这条线要做的事）

按优先级（同 `THEORY-BONNET.md` §5）：

- [ ] **跑那个归因实验阶梯**（§5 的 rung 1–5），重点是 **rung 4 = 一阶法向监督**
  这个真对手必须在。这是"非平凡且未被占" → "真能撑论文"的唯一闸门。
- [ ] 把 `THEORY-BONNET.md` §3.1 的旋转法向反例做成一个**合成单元测试**：构造该配置，
  验证光度/thinness/法向平滑都小、而 `symmetry_residual`/`shape_relative` 大。这把
  "非平凡性"从纸面变成可复现的 demo（评审最爱）。
- [ ] 在 torus（有曲率）上测 rung 4 vs rung 5，证明"曲率感知可积分性 vs 曲率盲平滑"
  的差异（§3.2 第 1 点）——这是打一阶监督基线的关键场景。
- [ ] 多 seed（0/1/2）+ held-out PSNR 护栏，出 paired CI，别只看一次。
- [ ] 文献：定期重检 Q1/Q3（GeoSplat 系、二阶 GS）有没有新论文占坑；本核查是单次
  非穷尽。
- [ ] 若 rung 4 已经把差距补平 → 诚实回退到守恒测度 `q_i` 主线，别硬撑 Bonnet。

---

## 一句话自检

学完你应该能回答这句话——如果能，这条线你就真的懂了：

> "3DGS 把位置和法向拆成两组自由变量；真实曲面里法向是位置的 Gauss 映射，受 Gauss
> 和 Codazzi 两条相容方程约束。我们用位置（Monge 拟合）和法向各自独立算一个第二基本
> 形/shape operator，罚它们的不相容（shape 差 + 对称性 + Gauss + Codazzi）。Bonnet
> 定理保证：残差归零等价于这对场真的来自同一张曲面。这和'把法向监督到位置法向'不同——
> 后者是一阶、注入噪声目标；我们是二阶、无目标、且曲率感知。"
