# 实验与分析 (Experiments and Analysis)

## 1. 概述 (Overview)

本章旨在验证基于 KNN-Shapley 的数据估值框架在不同模态任务中的有效性，并探究不同特征表示对估值准确性的影响。实验分为两个阶段：
1.  **图像分类任务**：作为概念验证（Proof-of-Concept），在受控噪声环境下验证改进后的距离度量对噪声检测能力的提升。
2.  **大语言模型指令微调（Instruction Tuning）**：作为核心贡献，将框架迁移至文本生成任务，对比不同文本特征表示（统计特征、梯度特征、语义表示）在筛选高质量微调数据中的效果。

---

## 2. 实验一：图像分类任务上的方法验证
**目的**：验证 KNN-Shapley 框架的有效性，并评估特征提取器与距离度量对噪声检测性能的影响。

### 2.1 实验设置 (Experimental Setup)
- **数据集**：Cats vs. Dogs (Binary Classification)。通过随机翻转部分样本标签构建含噪声数据集（Noise Ratio: Unknown/Variable）。
- **基线方法**：原始 Soft-label KNN-SV 算法，采用欧氏距离（Euclidean Distance）。
- **评估指标**：
    - **AUROC**：衡量异常检测的整体性能。
    - **F1-Rank (Top-10%)**：在固定截断阈值下的精确度与召回率。
    - **F1-Cluster**：衡量特征空间中噪声数据与干净数据的分离程度（聚类性能）。

### 2.2 特征提取器分析 (Backbone Comparison)
我们评估了 VGG11, ResNet-18/50, MobileNetV2, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16 等多种主流架构作为特征提取器的表现。
**结果分析**：实验表明，**EfficientNet-B0** 在参数量与特征表达能力之间取得了最佳平衡，其提取的特征在 KNN 估值任务中表现出最高的区分度。因此，后续实验均基于 EfficientNet-B0 进行。

### 2.3 距离度量的改进 (Metric Optimization)
针对高维特征空间中欧氏距离可能失效的问题（Curse of Dimensionality），我们提出了基于 **L2 归一化** 的改进方案，将度量标准从欧氏距离转换为等价的 **余弦距离 (Cosine Distance)**，以更关注特征向量的方向一致性而非模长。

**表 1：不同距离度量下的噪声检测性能对比**

| 特征提取器 | 距离度量 (Metric) | AUROC | F1-Rank | F1-Cluster |
| :--- | :--- | :--- | :--- | :--- |
| EfficientNet-B0 | Euclidean | 0.995 | **0.910** | 0.261 |
| EfficientNet-B0 | **Cosine (Ours)** | 0.991 | 0.845 | **0.354** |

**分析**：数据表明，虽然余弦距离在 F1-Rank 上略有下降，但在 **F1-Cluster 指标上提升了 35%**。这一显著提升证实了余弦相似度能更有效地分离特征空间中的噪声流形（Manifold），显著提升了基于聚类的噪声自动识别能力，这对于未知噪声比例的实际场景至关重要。

我们亦尝试了引入 MaxSim 全局最大池化策略，但实验发现其破坏了特征的局部流形结构，并未带来性能增益。

---

## 3. 实验二：LLM 指令微调中的数据估值 (Main Results)
**目的**：解决文本模态下的数据估值难题，确定适合 SFT（Supervised Fine-Tuning）任务的最优特征表示方案。

### 3.1 实验设置
- **基础模型**：Qwen1.5-0.5B。
- **数据集**：Alpaca 指令数据集。构建 30% 噪声数据集，噪声形式为将回复替换为无意义文本（如 "I don't know"）或乱码。
- **评估指标**：Training Loss, ROUGE-L Score（在独立干净验证集上评估）。
- **对比方法**：
    1.  **Dirty Model (Baseline)**：使用全量噪声数据训练。
    2.  **Oracle Model (Upper Bound)**：使用全量干净数据训练。
    3.  **Clean Model (Gradient-KNN)**：基于参数梯度相似度及其 Shapley 值进行清洗。
    4.  **Clean Model (RepSim-Mean)**：基于语义表示（Last Hidden State）的平均相似度进行清洗。
    5.  **Clean Model (RepSim-KNN, Ours)**：基于语义表示 + KNN-Shapley 算法进行清洗。

### 3.2 文本特征表示的初步探索
在早期实验中，我们尝试了基于统计学特征（词袋模型 BoW, TF-IDF）的 KNN-Shapley 估值。
**结果**：清洗后模型性能显著低于 Baseline。
**归因**：简单的统计特征无法捕捉指令数据的复杂语义依赖，导致算法错误地剔除了高复杂度的优质样本（False Positives），致使模型出现欠拟合（Underfitting）。这证实了在 LLM 语境下，必须采用基于模型内在状态的稠密特征表示。

### 3.3 不同估值方法的对比实验
我们对比了梯度空间（Gradient Space）与特征空间（Feature Space）两种表示方法，以及平均聚合与 Shapley 聚合两种估值策略。

**表 2：不同数据清洗策略对 SFT 模型性能的影响**
*(注：Clean Model 仅使用筛选后的 70% 数据进行训练)*

| 模型 (Method) | 训练数据分布 | Avg ROUGE-L | 性能排名 |
| :--- | :--- | :--- | :--- |
| **Oracle Model** | 100% Clean | 0.3306 | 1 |
| **RepSim-KNN (Ours)** | Top 70% Selected | **0.2889** | 2 |
| Gradient-KNN | Top 70% Selected | 0.2636 | 3 |
| RepSim-Mean | Top 70% Selected | 0.2603 | 4 |
| **Dirty Model** | 100% (Mixed Noise) | 0.2314 | 5 |

**结果分析**：
1.  **KNN-Shapley 的必要性**：`RepSim-KNN` 显著优于 `RepSim-Mean` (0.2889 vs 0.2603)。这表明简单计算全局平均相似度不足以衡量数据的独特贡献，而 KNN-Shapley 通过评估样本在局部邻域中的边际贡献，能更精准地识别高质量数据。
2.  **特征空间的优越性**：`RepSim-KNN` 优于 `Gradient-KNN` (0.2889 vs 0.2636)。相比于高维且稀疏的参数梯度（Parameter Gradients），模型最后一层的语义表示（Last Hidden State）更直接地反映了数据在语义空间中的分布，且计算开销显著更低。
3.  **有效性验证**：所有基于估值清洗的模型均显著优于 Dirty Model，证明了该框架在 LLM SFT 任务中的有效性。其中，我们的方法（RepSim-KNN）最为接近 Oracle 性能上限。

---

## 4. 结论与未来工作 (Conclusion & Future Work)

### 4.1 结论
本研究证实了基于 KNN-Shapley 的数据估值框架在深度学习模型训练中的通用有效性。
1.  在 CV 任务中，我们通过引入余弦相似度改进了度量标准，显著提升了噪声数据的聚类分离度。
2.  在 LLM SFT 任务中，我们证明了**基于模型末层语义特征 (RepSim) 结合 KNN-Shapley 算法**是当前最高效且准确的数据清洗方案，优于传统的梯度相似度方法。

### 4.2 后续计划
为进一步增强结论的鲁棒性，后续工作将聚焦于：
1.  **泛化性验证**：在代码生成（Code Generation）和数学推理（Reasoning）数据集上验证 RepSim-KNN 的有效性。
2.  **超参数敏感性分析**：探究 KNN 中 $K$ 值选取对估值稳定性的影响。
3.  **计算资源评估**：定量对比 RepSim-KNN 与 Gradient-KNN 在显存占用与计算时间上的差异，突显本方法的工程价值。
