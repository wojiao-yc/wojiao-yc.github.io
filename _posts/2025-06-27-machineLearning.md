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
[线性回归(linear regression)](#Linear-Regression) [支持向量机(Support Vector Machine)](#Support-Vector-Machine)



## 介绍
机器学习算法大致可以分为以下三类：

**监督学习算法（Supervised Algorithms）**:在监督学习训练过程中，可以由训练数据集学到或建立一个模式（函数 / learning model），并依此模式推测新的实例。通过已标注的数据（输入-输出对）学习映射关系，用于预测或分类。如，线性回归、逻辑回归、决策树、支持向量机。

**无监督学习算法 (Unsupervised Algorithms)**:这类算法从无标注数据中发现隐藏模式或结构，没有特定的目标输出，一般将数据集分为不同的组。如，K均值聚类、层次聚类、主成分分析、自编码器。

**强化学习算法 (Reinforcement Algorithms)**:通过与环境的交互学习最优策略，以最大化长期奖励。算法根据输出结果（决策）的成功或错误来训练自己，通过大量经验训练优化后的算法将能够给出较好的预测。类似有机体在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。如，Q-Learning、深度Q网络（DQN）、策略梯度（Policy Gradient）
---

## Linear Regression

目标是找到一条直线（或超平面）使得预测值与真实值之间的误差最小。其又分为两种类型，即只有一个自变量的简单线性回归(simple linear regression)与至少两组以上自变量的多变量回归(multiple regression)。

- **简单一元线性回归**

一元线性方程的公式是简单的: 
\[
y = wx + b
\]
给定两个点,就能确定其中的参数 $w$ 和 $b$ 。之后,就能在直角坐标系中画出相应的一条直线来。有了这条线,随意扔给你一个 $x$ ,你都能够通过这个函数式子求出 $y$ 的值。统计不像数学那么精确,统计会将理论与实际间的差别表示出来,也就是“误差”。因此,统计世界中的公式会有一个小尾巴 $\epsilon$ ,用来代表误差,即: $y = w_0 + w_1x + \epsilon$ 


[回到目录](#目录)

---

## Support Vector Machine