# 扩散人脸恢复论文中 CFG（Classifier-Free Guidance）的使用方式分析

## 概述

Classifier-Free Guidance（CFG，无分类器引导）是扩散模型中用于增强条件控制力的核心技术，由 Ho & Salimans 在 2022 年提出。在人脸恢复任务中，CFG 被广泛用于平衡**生成真实感**与**输入保真度**之间的关系。本文档系统分析多篇扩散人脸恢复论文中 CFG 的不同使用方式。

---

## 一、CFG 基本原理

标准 CFG 公式（以 Stable Diffusion 为例）：

$$
\tilde{\epsilon} = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \big( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \big)
$$

- $\epsilon_\theta(z_t, t, c)$：有条件预测（给定条件 $c$）
- $\epsilon_\theta(z_t, t, \emptyset)$：无条件预测（空条件）
- $w$：引导尺度（CFG Scale），$w > 1$ 增强条件影响

**核心思想**：推理时同时进行"有条件"和"无条件"两次预测，通过外推（extrapolation）放大条件的影响。

**训练前提**：模型必须在训练时以一定概率 dropout 条件（即输入空条件），使模型学会无条件生成能力。

---

## 二、各论文 CFG 使用方式对比

| 论文 | 会议/年份 | 引导类型 | CFG 公式形式 | 条件类型 | 条件 Dropout 策略 | 引导尺度 |
|------|----------|---------|-------------|---------|-----------------|---------|
| **FaceMe** | AAAI 2025 | **标准 CFG** | $\bar{z}_{t-1} = z_{t-1} + \lambda_{cfg} \times (z_{t-1}^{id} - z_{t-1})$ | 身份编码器特征 $c_{id}$ | 50% 替换 $c_{id} \to c_{text}$ | $\lambda_{cfg}$（可调超参） |
| **PGDiff** | NeurIPS 2023 | **Classifier Guidance**（非 CFG） | $p(x_{t-1}\|x_t, y) \approx \mathcal{N}(\mu_\theta + \Sigma_\theta g, \Sigma_\theta)$ | 属性分类器梯度 $g = \nabla \log p_\phi(y\|x)$ | 不适用（无需训练） | 动态梯度缩放 $s_{norm}$ |
| **Arc2Face** | ECCV 2024 | **标准 SD-CFG** | $\tilde{\epsilon} = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c_{id}) - \epsilon_\theta(z_t, t, \emptyset))$ | ArcFace 嵌入投射到 CLIP 空间 | 隐含在 SD 预训练中 | 标准 CFG Scale |
| **ReF-LDM** | NeurIPS 2024 | **不使用 CFG** | 无引导外推公式 | LQ 图像 + 参考图 CacheKV | 不适用 | 无 |
| **DifFace** | TPAMI 2024 | **不使用 CFG** | 基于过渡分布的 Markov 链去噪 | LQ 图像作为起始状态 | 不适用 | 无 |
| **OSEDiff** | NeurIPS 2024 | **不使用 CFG** | 单步 LQ→HQ 映射 + VSD 蒸馏 | LQ 图像 + 文本提示 | 不适用 | 无 |

---

## 三、各论文详细分析

### 1. FaceMe（AAAI 2025）—— 最典型的 CFG 应用

**论文**：FaceMe: Robust Blind Face Restoration with Personal Identification

**CFG 使用方式**：身份引导的个性化人脸修复

**训练策略**：
- 两阶段训练，第二阶段以 **50% 概率**将身份嵌入 $c_{id}$ 替换为普通文本嵌入 $c_{text}$（"a photo of a face"）
- 这一设计同时实现三个目的：
  1. **满足 CFG 数学前提**：模型学会无身份条件下的生成能力
  2. **平衡依赖**：防止模型过度依赖参考图而忽略低质量原图结构
  3. **解耦属性**：避免参考图的姿态/表情污染修复结果

**推理公式**：

$$
z_{t-1}^{id} = \phi(z_t, z_{LQ}, c_{id}), \quad z_{t-1} = \phi(z_t, z_{LQ})
$$

$$
\bar{z}_{t-1} = z_{t-1} + \lambda_{cfg} \times (z_{t-1}^{id} - z_{t-1})
$$

其中 $\phi(\cdot)$ 为模型，$z_{LQ}$ 为低质量图像潜变量，$\lambda_{cfg}$ 为控制身份强度的超参数。

**关键创新**：
- 在去噪的每一步进行**双路预测**（有/无身份），然后用 CFG 放大身份影响
- 结合小波色彩校正缓解 CFG 可能导致的色彩偏移

---

### 2. PGDiff（NeurIPS 2023）—— 采用 Classifier Guidance 而非 CFG

**论文**：PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance

**引导类型**：Classifier Guidance（分类器引导）

**核心思想**：不对退化过程建模，而是对高质量图像的**期望属性**建模

**与 CFG 的区别**：

| 特性 | Classifier Guidance | Classifier-Free Guidance (CFG) |
|------|-------------------|-------------------------------|
| 是否需要额外分类器 | ✅ 是 | ❌ 否 |
| 训练成本 | 无需重新训练扩散模型 | 需要训练时 dropout 条件 |
| 引导来源 | 分类器梯度 $\nabla \log p_\phi(y\|x)$ | 有/无条件预测的差值 |
| 灵活性 | 可组合多种属性 | 依赖单一条件向量 |

**PGDiff 的动态引导机制**：

1. **动态梯度缩放**：
   $$
   s_{norm} = \frac{\|x_t - x_{t-1}'\|_2}{\|g\|_2} \cdot s
   $$
   其中 $x_{t-1}' \sim \mathcal{N}(\mu_\theta, \Sigma_\theta)$

2. **多步梯度更新**：每个去噪步可执行多次梯度更新，而非传统的一次

3. **复合引导**：通过加权求和组合多个属性的梯度，支持多任务（如旧照片修复 = 修复 + 补全 + 上色）

---

### 3. Arc2Face（ECCV 2024）—— 继承 SD 的标准 CFG

**论文**：Arc2Face: A Foundation Model for ID-Consistent Human Faces

**CFG 使用方式**：直接继承 Stable Diffusion 的标准 CFG 机制

**条件注入**：
- 将 ArcFace 提取的 512 维身份嵌入，通过微调的 CLIP 文本编码器投射到 SD 的条件空间
- 完全放弃文本提示，仅依赖身份嵌入引导生成

**CFG 公式**（标准 SD 形式）：

$$
\tilde{\epsilon} = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c_{id}) - \epsilon_\theta(z_t, t, \emptyset))
$$

**特点**：
- 未对 CFG 机制本身做创新修改
- 核心创新在条件注入方式（ID → CLIP 空间映射）
- 身份编码器在训练中被冻结，依赖 SD 原有的条件 dropout 训练策略

---

### 4. ReF-LDM（NeurIPS 2024）—— 不使用 CFG

**论文**：ReF-LDM: A Latent Diffusion Model for Reference-based Face Image Restoration

**条件注入方式**：CacheKV 机制

**不使用 CFG 的原因**：
- 参考图通过 CacheKV 直接在自注意力层注入特征
- 使用 **timestep-scaled identity loss** 约束身份一致性，而非推理时的引导
- 模型天然学习在有无参考图时的自适应表现

**关键设计**：
- 参考图 KV 特征只需提取一次（t=0），在主去噪过程中重复使用
- 无显式的有条件/无条件双路预测
- 训练时不使用 condition dropout

---

### 5. DifFace（TPAMI 2024）—— 不使用 CFG

**论文**：DifFace: Blind Face Restoration With Diffused Error Contraction

**不使用 CFG 的原因**：
- 基于**过渡分布**的 Markov 链设计，直接从 LQ 图像逐步过渡到 HQ 图像
- 仅使用 L1 损失训练，利用预训练扩散模型的先验隐式正则化
- 不需要条件引导机制

**关键设计**：
- 从 LQ 图像到中间状态建立过渡分布
- 递归应用预训练扩散模型实现错误收缩（Error Contraction）
- 无需复杂的损失函数设计或多任务学习

---

### 6. OSEDiff（NeurIPS 2024）—— 不使用 CFG

**论文**：One-Step Effective Diffusion Network for Real-World Image Super-Resolution

**不使用 CFG 的原因**：
- **单步设计**：直接将 LQ 图像编码作为扩散起点，无需多步去噪
- 使用 **VSD（Variational Score Distillation）** 替代 CFG 作为正则化
- 从随机高斯噪声开始的多步扩散被完全抛弃

**关键设计**：
- 一步映射 $\hat{z}_H = \frac{z_L - \beta_T \epsilon_\theta(z_L; T, c_y)}{\alpha_T}$
- Latent-space VSD 使单步生成质量逼近多步扩散
- 仅需 LoRA 微调（8.5M 参数）

---

## 四、CFG 使用模式总结

### 模式一：标准 CFG（FaceMe, Arc2Face）

```
训练时: condition dropout → 模型学会无条件生成
推理时: 双路预测 (cond / uncond) → CFG 外推放大条件影响
```

**适用场景**：需要强条件控制（如身份保持）

**关键参数**：CFG scale $w$ 或 $\lambda_{cfg}$

**优势**：
- 无需额外分类器
- 引导强度连续可调
- 实现简单

**劣势**：
- 推理成本翻倍（需两次前向）
- 过大尺度会导致色彩偏移或伪影
- 需要训练时的 condition dropout

---

### 模式二：Classifier Guidance（PGDiff）

```
推理时: 属性分类器 → 梯度反传 → 修正去噪方向
```

**适用场景**：多任务、零样本适应

**关键参数**：梯度步数、动态缩放因子

**优势**：
- 无需重新训练模型
- 可组合多种属性引导
- 适用于零样本场景

**劣势**：
- 需要额外分类器
- 梯度反传计算开销大
- 分类器质量直接影响引导效果

---

### 模式三：不使用显式引导（ReF-LDM, DifFace, OSEDiff）

```
训练时: 隐式学习条件与无条件表现
推理时: 单次前向 → 直接输出
```

**适用场景**：追求效率、简化流程

**优势**：
- 推理速度快（单次前向）
- 无需超参数调节（无 CFG scale）
- 训练简单

**劣势**：
- 条件控制强度不可在推理时动态调节
- 灵活性较低

---

## 五、CFG 参数对人脸恢复的影响

| 参数 | 影响 | 典型设置 |
|------|------|---------|
| CFG scale 过小 ($w \approx 1$) | 接近无条件生成，身份保持弱 | - |
| CFG scale 适中 ($w = 3{\sim}7$) | 良好的引导效果 | FaceMe 使用 $\lambda_{cfg}$ |
| CFG scale 过大 ($w > 10$) | 色彩偏移、过度锐化、不自然伪影 | 需配合色彩校正 |
| Condition dropout 概率 | 控制模型对条件的依赖程度 | FaceMe 使用 50% |
| 参考图数量 | 影响身份特征质量 | FaceMe 支持 1∼4 张 |

---

## 六、总结与趋势

1. **CFG 是扩散模型人脸恢复中最主流的引导方式**，尤其在需要强身份保持的场景（如 FaceMe）。

2. **Classifier Guidance**（如 PGDiff）在零样本多任务场景中具有独特优势，但计算成本较高。

3. **最新趋势**（OSEDiff 为代表）正尝试不依赖 CFG，通过单步设计和 VSD 蒸馏来兼顾效率和质量。

4. **CFG 的具体实现因任务而异**：
   - 纯生成任务（Arc2Face）：使用标准 CFG
   - 身份保持修复（FaceMe）：CFG + 双路身份预测
   - 通用修复（PGDiff）：Classifier Guidance + 动态引导
   - 效率优先（OSEDiff）：放弃 CFG，使用 VSD

5. **色彩校正常常是 CFG 的必要配套**，因为高 CFG scale 容易导致色彩偏移（FaceMe 使用小波色彩校正）。

---

*分析日期：2026-05-26*
*分析范围：face-restoration&SR 目录下的扩散人脸恢复论文*
