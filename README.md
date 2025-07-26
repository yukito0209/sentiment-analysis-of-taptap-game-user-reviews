# TapTap游戏用户评论情感分析

<!-- 基于机器学习的TapTap游戏用户评论情感分析研究 -->

一个全面的情感分析项目，针对TapTap移动游戏平台的中文用户评论，运用从传统机器学习到深度学习的多种技术进行情感分类。

## 项目概述

本项目旨在开发和评估各种机器学习模型，用于分析TapTap游戏平台用户评论的情感倾向。通过对比从词典方法到先进集成学习的多种技术，实现准确的正负面情感分类。

**最终模型准确率：86%**（通过堆叠集成模型实现）

## 核心特性

- **数据收集**：自定义爬虫工具，从TapTap API获取用户评论
- **全面预处理**：处理非正式文本、表情符号、中文NLP特殊性
- **模型对比**：涵盖词典方法、传统ML、梯度提升、深度学习和预训练语言模型
- **先进集成**：堆叠泛化模型，结合XGBoost、CatBoost和BERT-base-Chinese
- **详细分析**：完整的性能指标和可视化结果

## 项目结构

```
sentiment-analysis-of-taptap-game-user-reviews/
├── analytics/                          # 核心分析代码
│   ├── data_cleaning/                   # 数据清洗
│   │   ├── data_cleaning.py            # 主要数据清洗脚本
│   │   └── integrate.ipynb             # 数据整合笔记本
│   ├── data_exploring/                  # 数据探索
│   │   ├── data_exploring.ipynb        # 探索性数据分析
│   │   └── taptap_wordcloud_high_res.png
│   ├── traditional_ml_models/           # 传统机器学习模型
│   │   ├── decision_tree.ipynb         # 决策树
│   │   ├── knn.ipynb                   # K近邻
│   │   ├── logistic_reg.ipynb          # 逻辑回归
│   │   └── svm.ipynb                   # 支持向量机
│   ├── ensemble_learning_models/        # 集成学习模型
│   │   ├── adaboost.ipynb              # AdaBoost
│   │   ├── catboost.ipynb              # CatBoost
│   │   └── xgboost.ipynb               # XGBoost
│   ├── neural_network_models/           # 神经网络模型
│   │   ├── bilstm.ipynb                # 双向LSTM
│   │   └── cnn.ipynb                   # 卷积神经网络
│   ├── pretrained_language_models/      # 预训练语言模型
│   │   ├── bert-base-chinese.ipynb     # BERT中文基础模型
│   │   ├── chinese-roberta-wwm-ext.ipynb # RoBERTa中文模型
│   │   ├── Erlangshen-Roberta-110M-Sentiment.ipynb
│   │   ├── gpt2-chinese-cluecorpussmall.ipynb
│   │   └── t5-base.ipynb
│   ├── ensemble_voting/                 # 集成投票方法
│   │   └── stacking.ipynb              # 堆叠泛化
│   ├── lexicon/                        # 词典方法
│   │   └── snownlp.py                  # SnowNLP情感分析
│   └── stopwords/                      # 停用词
│       └── stopwords_hit.txt
├── data collection/                     # 数据收集
│   └── get_taptap_reviews.py           # TapTap评论爬虫
├── data/                               # 数据文件
│   └── README.md                       # 数据集说明
├── visualisation/                       # 可视化结果
│   ├── visualisation.ipynb            # 可视化代码
│   ├── model_accuracy_comparison_high_res.png
│   ├── *_confusion_matrix.png          # 各模型混淆矩阵
│   ├── *_classification_report.png     # 各模型分类报告
│   └── Stacked_Generalization_Diagram_Final_v2.png
├── LICENSE
└── README.md
```

## 数据集说明

- **数据源**：TapTap平台公开用户评论 (taptap.com / taptap.io)
- **收集方式**：自定义Python爬虫，模拟浏览器请求TapTap API
- **数据规模**：约40,000条评论，来自40款热门游戏（每款游戏1000条最新评论）
- **原始特征**：用户ID、用户名、评分(1-5)、评论内容、点赞数、发布时间、设备型号、游戏名称
- **目标变量**：情感二分类（0=负面[1-2星]，1=正面[3-5星]）
- **处理后数据**：清洗后包含39,985条有效评论

**数据获取**：完整数据集已上传至Kaggle - [TapTap Mobile Game Reviews (Chinese)](https://www.kaggle.com/datasets/karwinwang/taptap-mobile-game-reviews-chinese)

## 研究方法

### 1. 数据收集与预处理
- **爬虫开发**：模拟浏览器请求，处理反爬虫机制
- **数据清洗**：处理缺失值、HTML标签、表情符号、特殊字符
- **中文处理**：jieba分词、停用词过滤、文本标准化

### 2. 特征工程
- **传统ML模型**：TF-IDF向量化
- **深度学习模型**：词嵌入和序列编码
- **预训练模型**：BERT tokenizer
- **增强特征**：游戏名称、点赞数等元数据特征

### 3. 模型评估
#### 基线模型
- **词典方法**：SnowNLP情感分析

#### 传统机器学习
- 决策树、K近邻、逻辑回归、支持向量机

#### 集成学习
- AdaBoost、XGBoost、CatBoost

#### 深度学习
- 卷积神经网络(CNN)、双向长短期记忆网络(BiLSTM)

#### 预训练语言模型
- BERT-base-Chinese、RoBERTa-wwm-ext、GPT2-Chinese等

#### 集成方法
- **堆叠泛化**：
  - 基学习器（Level 0）：XGBoost、CatBoost、BERT-base-Chinese
  - 元学习器（Level 1）：逻辑回归

## 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.0+（可选，用于GPU加速）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yukito0209/sentiment-analysis-of-taptap-game-user-reviews.git
cd sentiment-analysis-of-taptap-game-user-reviews
```

2. **创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
# 基础依赖
pip install pandas numpy scikit-learn matplotlib seaborn jieba

# 深度学习依赖
pip install torch torchvision transformers

# 梯度提升模型
pip install xgboost catboost

# 其他工具
pip install requests beautifulsoup4 emoji snownlp
```

### 使用方法

#### 数据收集
```bash
# 修改游戏ID和输出文件名
python "data collection/get_taptap_reviews.py"
```

#### 数据清洗
```bash
cd analytics/data_cleaning
python data_cleaning.py
```

#### 模型训练与评估
```bash
# 在对应的目录中运行Jupyter Notebook
jupyter notebook

# 按以下顺序运行：
# 1. analytics/data_exploring/data_exploring.ipynb - 数据探索
# 2. analytics/traditional_ml_models/ - 传统ML模型
# 3. analytics/ensemble_learning_models/ - 集成学习模型
# 4. analytics/neural_network_models/ - 深度学习模型
# 5. analytics/pretrained_language_models/ - 预训练模型
# 6. analytics/ensemble_voting/stacking.ipynb - 堆叠集成
# 7. visualisation/visualisation.ipynb - 结果可视化
```

## 实验结果

### 模型性能对比

| 模型类别 | 模型名称 | 准确率 |
|---------|---------|--------|
| 词典方法 | SnowNLP | 67% |
| 传统ML | 决策树 | 72% |
|  | K近邻 | 75% |
|  | AdaBoost | 77% |
|  | 逻辑回归 | 81% |
|  | SVM | 81% |
|  | XGBoost | 83% |
|  | CatBoost | 83% |
| 深度学习 | CNN | 79% |
|  | BiLSTM | 80% |
| 预训练模型 | GPT2-Chinese | 83% |
|  | BERT-base-Chinese | 84% |
|  | RoBERTa-wwm-ext | 84% |
| **集成方法** | **堆叠泛化** | **86%** |

### 关键发现
- **堆叠集成模型**达到最佳性能（86%准确率）
- **BERT类模型**在单模型中表现最优（84%准确率）
- **模型组合**显著提升了分类性能
- **中文预训练模型**比传统方法更适合中文情感分析

详细的性能指标和混淆矩阵可在`visualisation/`目录中查看。

## 未来改进方向

- **错误分析**：深入分析误分类样本
- **中性评论处理**：探索3星评论的标注策略
- **元学习器优化**：尝试更复杂的元学习算法
- **数据增强**：提高模型鲁棒性的技术
- **文本预处理增强**：更好地处理网络用语和错字
- **实时部署**：开发在线情感分析API

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
