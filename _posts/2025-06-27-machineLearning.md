---
date: 2025-06-27 18:00:00
layout: post
title: Machine Learning Summary
subtitle: Recording and Learning
description: Summarize and memorize the content of machine learning
image: ..\assets\img\posts\MachineLearning\machineLearning.png
optimized_image: ..\assets\img\posts\MachineLearning\machineLearning.png
category: study
tags:
  - machine learning
author: wojiao-yc
---

## 目录
### [介绍](#介绍)

### 经典算法
[线性回归(linear regression)](#线性回归) &nbsp;&nbsp;&nbsp; [支持向量机(Support Vector Machine)](#支持向量机) &nbsp;&nbsp;&nbsp; [k-近邻(K-Nearest Neighbors)](#k-近邻) &nbsp;&nbsp;&nbsp; [逻辑回归(Logistic Regression)](#逻辑回归) &nbsp;&nbsp;&nbsp; [决策树(Decision Tree)](#决策树) &nbsp;&nbsp;&nbsp; [k-平均(k-means)](#k-平均) &nbsp;&nbsp;&nbsp; [随机森林(Random Forest)](#随机森林) &nbsp;&nbsp;&nbsp; [朴素贝叶斯(Naive Bayes)](#朴素贝叶斯)

### 损失函数

### 激活函数

### 其它


## 介绍
机器学习算法大致可以分为以下三类：

**监督学习算法（Supervised Algorithms）**:在监督学习训练过程中，可以由训练数据集学到或建立一个模式（函数 / learning model），并依此模式推测新的实例。通过已标注的数据（输入-输出对）学习映射关系，用于预测或分类。如，线性回归、逻辑回归、决策树、支持向量机。

**无监督学习算法 (Unsupervised Algorithms)**:这类算法从无标注数据中发现隐藏模式或结构，没有特定的目标输出，一般将数据集分为不同的组。如，K均值聚类、层次聚类、主成分分析、自编码器。

**强化学习算法 (Reinforcement Algorithms)**:通过与环境的交互学习最优策略，以最大化长期奖励。算法根据输出结果（决策）的成功或错误来训练自己，通过大量经验训练优化后的算法将能够给出较好的预测。类似有机体在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。如，Q-Learning、深度Q网络（DQN）、策略梯度（Policy Gradient）。

---

## 线性回归

### 定义与核心思想
一种用于建模连续变量之间关系的**监督学习**算法。其核心假设是目标变量与特征之间存在线性关系，并通过拟合最佳直线（或超平面）进行预测，使得预测值与真实值之间的误差最小。其又分为两种类型，即只有一个自变量的简单线性回归(simple linear regression)与至少两组以上自变量的多变量回归(multiple regression)。

### 目标
找到最优的权重 $w$ 和偏置 $b$，使得预测值与真实值之间的均方误差（MSE）最小化。

### 详细内容
推荐看这篇[知乎专栏](https://zhuanlan.zhihu.com/p/72513104)。

### 实现流程
1. 初始化模型参数（权重w和偏置b），通常设为0或小的随机值
2. 计算预测值并计算损失函数（通常使用均方误差MSE）
3. 计算损失函数对参数的梯度
4. 使用梯度下降法更新参数

### 代码实现
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression model
        
        Parameters:
        learning_rate -- step size for parameter updates (default 0.01)
        n_iterations -- number of training iterations (default 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # weight parameters
        self.bias = None     # bias parameter
        self.loss_history = []  # to record loss at each iteration
    
    def fit(self, X, y):
        """
        Train the linear regression model
        
        Parameters:
        X -- feature matrix of shape (n_samples, n_features)
        y -- target vector of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize parameters
        self.weights = np.zeros(n_features)  # initialize weights to 0
        self.bias = 0                       # initialize bias to 0
        
        # 2. Gradient descent iterations
        for _ in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss (mean squared error)
            loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # weight gradients
            db = (1 / n_samples) * np.sum(y_pred - y)         # bias gradient
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        X -- feature matrix of shape (n_samples, n_features)
        
        Returns:
        y_pred -- predicted values of shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
```


### 面经
**Q**：线性回归的基本假设有哪些？
> 线性关系假设（自变量X与因变量y之间存在线性关系），误差项独立同分布（无自相关），误差项之间无相关性（尤其时间序列数据），误差项正态分布（残差应服从均值为0的正态分布）

**Q**：如何判断线性回归模型的好坏？
> R²分数来解释模型的方差解释能力(0-1，越接近1越好)，均方误差(MSE)/均方根误差(RMSE)越小越好，残差分析（检查残差是否随机分布），也可以对比训练集和测试集表现来判断过拟合。

**Q**：什么是R²分数？
> 是评估线性回归模型拟合优度的指标，表示模型能够解释的目标变量方差比例。其取值范围通常在0到1之间，数值越大表示模型解释能力越强。R²通过比较模型预测误差（实际值与模型预测值的差异平方和）和基准误差（实际值与均值的差异平方和）来计算，具体来说计算公式为：R² = 1 - (模型预测误差 / 基准误差)

[回到目录](#目录)

---

## 支持向量机

### 定义与核心思想
支持向量机是一种强大的**监督学习**算法，主要用于分类任务，也可用于回归（称为支持向量回归，SVR）。对于线性可分的数据集来说，这样的超平面有无穷多个（即感知机），但是几何间隔最大的分离超平面却是唯一的。SVM还包括核技巧，即对于输入空间中的非线性分类问题，可以通过非线性变换将它转化为某个维特征空间中的线性分类问题，在高维特征空间中学习线性支持向量机，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题。总的来说，其核心思想是寻找一个最优超平面，最大化不同类别数据之间的边界（间隔），从而提升模型的泛化能力。

### 目标
- 最大化分类间隔，即找到使间隔最大的超平面，提高模型的泛化能力。
- 最小化分类错误（在软间隔 SVM 中，允许少量样本违反间隔约束）。

### 详细内容
该[知乎专栏](https://www.zhihu.com/tardis/zm/art/31886934?source_id=1005)提供了极为详细的数学推导，这个[知乎专栏](https://zhuanlan.zhihu.com/p/77750026)则更为通俗易懂。

### 实现流程
1. 构造最优化问题，求解出最优化的所有α
2. 计算参数 $w$ 和 $b$
3. 得出超平面与决策函数

### 代码实现

### 面经
**Q**：SVM为什么追求“最大间隔”？
> 最大间隔能提高模型的泛化能力，使决策边界对噪声和数据扰动更鲁棒。间隔越大，未来新数据被分类错误的可能性越小。

**Q**：SVM的优点是什么？
> 1.**在高维空间中表现优秀**：训练好的模型的算法复杂度是由支持向量的个数决定的，而不是由数据的维度决定的，所以SVM也不太容易产生overfitting
 2.**核技巧灵活**，能适应复杂非线性问题：通过核函数，SVM可以隐式地将数据映射到更高维空间，从而解决线性不可分问题
 3.依赖支持向量而非全部数据，**内存效率高**：训练完成后，可以丢弃非支持向量的样本，节省存储空间，适合资源受限的场景

**Q**：SVM的缺点是什么？
> 1.**对参数和核函数选择敏感**：惩罚系数 $C$ 控制模型对分类错误的容忍度，越大说明越不能容忍出现误差，容易过拟合；越小则越容易欠拟合。另外不同的核函数对结果影响很大，需要交叉验证调整
2.**黑盒性较强**，可解释性差：相比逻辑回归能分析权重和决策树能可视化规则，SVM的决策过程较难解释，且无法直接输出概率（需额外校准，如Platt Scaling）。

**Q**：SVM为什么追求“最大间隔”？
> 最大间隔能提高模型的泛化能力，使决策边界对噪声和数据扰动更鲁棒。间隔越大，未来新数据被分类错误的可能性越小。

**Q**：SVM如何进行概率预测？
> 可以使用 Platt Scaling 进行校准，在SVM的输出（决策函数值）上训练一个逻辑回归模型，将其映射到 [0,1] 区间。

[回到目录](#目录)

---

## k-近邻

### 定义与核心思想
K-近邻 是一种基于样本的**监督学习**算法，可用于分类和回归任务。它的核心思想是：相似的数据点在特征空间中距离较近，因此新样本的类别或值可以由其最近的K个邻居决定。具体来说是给定一个训练数据集，对新的输入样本，在训练数据集中找到与该实例最邻近的K个样本，这K个样本的多数属于某个类，就把该输入样本呢分类到这个类中。（少数服从多数）

### 目标
- 分类任务：基于K个最近邻的多数投票，预测新样本的类别。
- 回归任务：基于K个最近邻的平均值，预测新样本的连续值。

### 详细内容
推荐看这篇[文章](https://blog.csdn.net/m0_74405427/article/details/133714384)。

### 实现流程
1. 距离计算：对于待分类的样本点，计算它与训练集中每个样本点的距离
2. 选择最近邻：根据计算的距离，选择距离最近的k个训练样本，即k-近邻名字的由来
3. 投票决策：对于分类问题统计k个最近邻中各类别的数量，将待分类样本归为数量最多的类别；对于回归问题：取k个最近邻的目标值的平均值作为预测值

### 代码实现
```python
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KNN:
    """
    K-Nearest Neighbors classifier manual implementation
    
    Parameters:
        k: int, optional (default=5), number of nearest neighbors to consider
    """
    
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        """
        Train the KNN model (actually just stores the data as KNN is a lazy learner)
        
        Parameters:
            X: Training feature data, array of shape [n_samples, n_features]
            y: Training target values, array of shape [n_samples]
        """
        # Standardize data (improves KNN performance)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y
        
    def predict(self, X):
        """
        Make predictions for test data
        
        Parameters:
            X: Test feature data, array of shape [n_samples, n_features]
            
        Returns:
            Predictions, array of shape [n_samples]
        """
        # Standardize test data (using mean and variance from training data)
        X = self.scaler.transform(X)
        # Initialize prediction array
        predictions = np.zeros(X.shape[0], dtype=self.y_train.dtype)
        
        # Make prediction for each test sample
        for i, x in enumerate(X):
            # 1. Calculate distances between current test sample and all training samples
            distances = self._compute_distances(x)
            
            # 2. Get indices of k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            
            # 3. Get labels of these k neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # 4. Vote for prediction (take most common class)
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions[i] = most_common[0][0]
            
        return predictions
    
    def _compute_distances(self, x):
        """
        Calculate distances between one sample and all training samples (Euclidean distance)
        
        Parameters:
            x: Single sample point, array of shape [n_features]
            
        Returns:
            Array of distances, array of shape [n_train_samples]
        """
        # Euclidean distance calculation: sqrt(sum((x1 - x2)^2))
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        return distances
```
### 面经
**Q**： 如何选择合适的K值？
>通过交叉验证（一种评估机器学习模型泛化能力的统计方法，核心思想是通过多次划分训练集和验证集，减少模型评估的随机性，从而更准确地估计模型在未知数据上的表现）尝试不同K值，选择在验证集上表现最好的K

**Q**：讲一下KNN的优点
>1.**简单直观，易于实现**:直接通过距离计算找到最近的邻居，无需复杂的数学推导
2.**无需训练阶段**:KNN仅存储训练数据，预测时才计算距离，无需显式训练参数
3.**对数据分布没有假设**：不像线性回归假设数据线性可分，或高斯朴素贝叶斯假设特征符合正态分布。KNN通过局部邻居投票，能捕捉非线性关系

**Q**：讲一下KNN的缺点
>1.计算复杂度高：需保存全部训练数据，内存占用大，对新样本需计算与所有训练样本的距离，时间复杂度为O(N·d)
2.高维数据效果差（维度灾难）：在高维空间中，所有样本的距离趋于相似（欧氏距离区分度下降），导致邻居失去意义
3.对不平衡数据敏感：若某类样本占90%，K个邻居中大概率全是多数类，少数类易被忽略
4.需要特征缩放：若特征尺度差异大（如年龄[0-100]和工资[0-100000]），工资会主导距离计算


[回到目录](#目录)

---

## 逻辑回归

### 定义与核心思想

逻辑回归是一种典型的**监督学习**算法，模型从带有标签的训练数据中学习规律，并对新数据进行预测。虽然被称为回归，但其实际上是分类模型，并常用于二分类。本质是假设数据服从某个分布，然后使用极大似然估计做参数的估计。其核心思想则是通过线性模型预测概率并用逻辑函数（Sigmoid函数）将线性结果映射到[0,1]区间。

### 目标

逻辑回归的目标是找到一组模型参数（权重$𝑤$和偏置$𝑏$），使得模型的预测概率尽可能接近真实的类别标签。这一目标通过最大化似然函数（或最小化损失函数）来实现。

### 实现流程

### 代码实现

### 面经
**Q**：逻辑回归和感知机的区别。
>简单的感知机其实和逻辑回归类似，都是数据乘上一个回归系数矩阵 $w$ 得到一个数 $y$，不过感知机不求概率，一般会选取一个分类边界，可能 $y>0$ 就是 $A$ 类别，$y<0$ 就是 $B$ 类别。逻辑回归的损失函数由最大似然推导而来，用交叉熵损失，力图使预测概率分布与真实概率分布接近。感知机的损失函数可能有多种方法核心是针对误分类点到超平面的距离总和进行建模，即使预测的结果与真实结果误差更小，是去求得分类超平面（函数拟合）。这是两者最最根本的差异。

[回到目录](#目录)

---


## 决策树

### 定义与核心思想

决策树是一种基于树结构的**监督学习**算法，主要用于分类和回归任务。它通过一系列规则对数据进行分割，构建一个树形模型，其中每个内部节点代表一个特征或属性上的判断条件，每个分支代表判断条件的可能结果，而每个叶节点代表一个类别（分类树）或一个具体值（回归树）。核心思想是递归地选择最优特征进行数据划分，使得划分后的子集尽可能“纯净”，即同一类别的样本尽可能集中。

### 目标
决策树的目标是构建一个泛化能力强、解释性好的模型，具体包括：
- 分类任务：叶节点表示类别标签，目标是最大化分类准确性。
- 回归任务：叶节点表示连续值，目标是最小化预测误差（如均方误差）。

此外，决策树追求：
- 局部最优性：每次划分选择当前最优特征，而非全局最优。
- 可解释性：通过树形结构直观展示决策逻辑（类似“if-then”规则）。

### 实现流程

### 代码实现

### 面经
**Q**：
>

[回到目录](#目录)

---


## k-平均

### 定义与核心思想
k-means 是一种经典的**无监督学习**聚类算法，用于将数据集划分为 k 个互不重叠的簇（clusters）。其目标是通过迭代优化，将数据点分配到最近的簇中心（质心），使得簇内数据点的相似性较高，而不同簇之间的差异性较大。

### 目标

k-means 的核心目标是将数据集划分为 k 个簇（clusters），使得每个数据点属于距离最近的簇中心。通过反复调整簇中心的位置，k-means 不断优化簇内的紧密度，从而获得尽量紧凑、彼此分离的簇。这可以用簇内平方误差（Within-Cluster Sum of Squares, WCSS）来度量：

$\text{WCSS} = \sum_{i=1}^{K} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \mathbf{\mu}_i\|^2$ 
 
- $K$: 预设的簇数量。  
- $C_i$: 第 $i$ 个簇的集合。  
- $\mathbf{\mu}_i$: 第 $i$ 个簇的质心（均值向量）。  
- $\mathbf{x}$: 数据点。  

### 实现流程

### 代码实现

### 面经
**Q**：
>

[回到目录](#目录)

---

## 随机森林

### 定义与核心思想
一种基于集成学习（Ensemble Learning）的**监督学习**算法，主要用于分类和回归任务。它通过构建多棵决策树（Decision Trees）并结合它们的预测结果来提高模型的准确性和鲁棒性。随机森林由多棵决策树组成，每棵树独立训练并投票（分类）或平均（回归），然后引入两种随机性（特征随机选择、数据随机采样）来增强多样性，防止过拟合。核心思想是 “集体智慧”，即多个弱模型组合成一个强模型

### 目标

- 最大化分类准确性或回归精度：通过多棵树的集体决策降低方差（Variance），提升模型稳定性。
- 最小化过拟合：利用随机采样和特征选择，确保每棵树差异较大，避免模型过于依赖训练数据中的噪声。

### 实现流程

### 代码实现

### 面经
**Q**：
>

[回到目录](#目录)

---


## 朴素贝叶斯

### 定义与核心思想
一种基于贝叶斯定理的**监督学习**算法，主要用于分类任务。核心思想是计算给定特征下样本属于某类的后验概率，并选择概率最大的类别作为预测结果。关键假设为所有特征相互独立，即“朴素”一词的由来，因此联合概率可拆分为单特征概率的乘积。

### 目标
朴素贝叶斯算法的优化目标是找到使后验概率 $P(y \mid X)$ 最大的类别 $y$，数学表达式为：

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^n P(x_i \mid y)
$$

- $P(y)$：类别 $y$ 的先验概率
- $P(x_i \mid y)$：特征 $x_i$ 在类别 $y$ 下的条件概率
- $\prod_{i=1}^n P(x_i \mid y)$：基于特征独立性假设的联合概率
- $\arg\max_{y}$：选择使后验概率最大的类别

### 详细内容
推荐看这篇[知乎专栏](https://zhuanlan.zhihu.com/p/656721603)。

### 实现流程
1. 训练阶段输入已标注样本集输出各类别先验和各特征条件概率，即统计每个类别出现的频率，以及对每个特征在每个类别下统计其频率
2. 预测阶段根据样本特征计算每个类别的概率，选择概率最大的类别作为预测类别

### 代码实现

```python
import numpy as np

class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes classifier implementation.
    Assumes features follow normal distribution and are conditionally independent given the class.
    """
    
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model with training data.
        
        Parameters:
        X : numpy array, shape (n_samples, n_features)
            Training feature vectors
        y : numpy array, shape (n_samples,)
            Target class labels
        """
        # Get unique class labels from training data
        self.classes = np.unique(y)
        
        # Initialize dictionaries to store:
        self.mean = {}      # Mean of each feature per class
        self.var = {}       # Variance of each feature per class
        self.priors = {}    # Prior probability of each class
        
        # Calculate statistics for each class
        for c in self.classes:
            # Get samples belonging to current class
            X_c = X[y == c]
            
            # Calculate mean and variance of each feature for this class
            self.mean[c] = X_c.mean(axis=0)  # Mean along each feature column
            self.var[c] = X_c.var(axis=0)   # Variance along each feature column
            
            # Calculate prior probability: P(class=c)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def predict(self, X):
        """
        Make predictions for new samples using the trained model.
        
        Parameters:
        X : numpy array, shape (n_samples, n_features)
            Test feature vectors to predict
            
        Returns:
        predictions : list
            Predicted class labels for each test sample
        """
        predictions = []
        
        # Process each test sample individually
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for c in self.classes:
                # Calculate likelihood using Gaussian PDF (simplified computation):
                # P(x|class=c) = product of P(x_i|class=c) for all features
                exponent = -((x - self.mean[c]) ** 2) / (2 * self.var[c])
                likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * self.var[c])
                
                # Calculate posterior: P(class=c|x) ∝ P(x|class=c) * P(class=c)
                posterior = np.prod(likelihood) * self.priors[c]
                posteriors.append(posterior)
            
            # Select class with highest posterior probability
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
            
        return predictions
```

### 面经
**Q**：为什么叫朴素贝叶斯？
>“朴素”指的是它对特征做了一个强假设认为所有特征之间完全独立。现实中这很少成立（比如“天气”和“湿度”可能相关），但这个假设简化了计算，使得算法高效。

**Q**：朴素贝叶斯优点？
>**计算快**：1.假设所有特征相互独立，计算联合概率时只需简单相乘，无需考虑特征间的复杂交互 2.训练时只需统计每个特征在各类别下的频率，预测时直接查表计算，速度极快
**对小数据友好**：1.只需估计每个特征的边缘概率，而非特征间的联合概率 2.参数少，不容易过拟合
**简单易实现**：1.多数实现只需选择分布类型，无需像SVM调核函数或随机森林调树深度 2.模型本质是一个概率统计表，训练过程只是计数或计算均值/方差，无需梯度下降等迭代优化
**对缺失数据不敏感**：1.计算概率时，若某特征缺失，直接跳过该特征的乘积项，不影响其他特征贡献 2.统计特征概率时，缺失值不参与计数。

**Q**：朴素贝叶斯缺点？
>**独立性假设太强**、**零概率问题**（模型对未见特征直接判零概率），以及**对输入分布敏感**（如果数据不符合假设，效果可能变差）。



[回到目录](#目录)

---


## 最小二乘法
最小二乘法是一种数学优化技术，常用于数据拟合问题，目标是找到一个函数，使其尽可能准确地逼近一组观测数据。

### 一、基本思想

最小二乘法的核心思想是：  
**在一组给定数据点下，选取一个函数，使得该函数预测值与实际值之间的“误差平方和”最小。**

假设有 $n$ 个观测数据点：

$(x_1, y_1),\ (x_2, y_2),\ \dots,\ (x_n, y_n)$

我们希望找到一个函数，比如线性函数：

$y = ax + b$

使得预测值 $\hat{y}_i = ax_i + b$ 与真实值 $y_i$ 的误差最小。误差平方和为：

$S(a, b) = \sum_{i=1}^n (y_i - (ax_i + b))^2$

目标是求：

$\min_{a, b} S(a, b)$

### 二、推导过程（线性拟合）

对 $S(a, b)$ 对参数 $a$ 和 $b$ 求偏导，并令偏导为 0，得到正规方程：

$\frac{\partial S}{\partial a} = -2 \sum_{i=1}^n x_i(y_i - ax_i - b) = 0$

$\frac{\partial S}{\partial b} = -2 \sum_{i=1}^n (y_i - ax_i - b) = 0$

整理得到：

$a \sum x_i^2 + b \sum x_i = \sum x_i y_i$

$a \sum x_i + nb = \sum y_i$

解这个二元一次方程组即可得到最优的 $a$ 和 $b$。

### 三、扩展形式
多元线性回归

$y = a_1x_1 + a_2x_2 + \dots + a_kx_k + b$

矩阵形式表达（适用于编程实现）

$\text{给定 } A\in\mathbb{R}^{n\times p},\ b\in\mathbb{R}^{n},\ \text{最小化 } \|Ax - b\|_2^2$

解为：

$x = (A^TA)^{-1}A^Tb \quad (\text{前提：} A^TA \text{ 可逆})$

[回到目录](#目录)

---
