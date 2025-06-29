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

**监督学习算法（Supervised Algorithms）**:在监督学习训练过程中，可以由训练数据集学到或建立一个模式（函数 / learning model），并依此模式推测新的实例。该算法要求特定的输入/输出，首先需要决定使用哪种数据作为范例。例如，文字识别应用中一个手写的字符，或一行手写文字。主要算法包括神经网络、支持向量机、最近邻居法、朴素贝叶斯法、决策树等。

**无监督学习算法 (Unsupervised Algorithms)**:这类算法没有特定的目标输出，算法将数据集分为不同的组。

**强化学习算法 (Reinforcement Algorithms)**:强化学习普适性强，主要基于决策进行训练，算法根据输出结果（决策）的成功或错误来训练自己，通过大量经验训练优化后的算法将能够给出较好的预测。类似有机体在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为

[回到目录](#目录)
---

## Linear Regression

目标是找到一条直线（或超平面）使得预测值与真实值之间的误差最小。其又分为两种类型，即只有一个自变量的简单线性回归(simple linear regression)与至少两组以上自变量的多变量回归(multiple regression)。

- 简单一元线性回归
一元线性方程的公式应该是简单的: 
\[
y = wx + b
\]
给定两个点,就能确定其中的参数 $w$ 和 $b$ 。之后,就能在直角坐标系中画出相应的一条直线来。有了这条线,随意扔给你一个 $x$ ,你都能够通过这个函数式子求出 $y$ 的值。


[回到目录](#目录)

---

## Support Vector Machine