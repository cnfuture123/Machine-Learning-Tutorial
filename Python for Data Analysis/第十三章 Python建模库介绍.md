# Python建模库介绍

## pandas与模型代码的接口

  - 模型开发的通常工作流是使用pandas进行数据加载和清洗，然后切换到建模库进行建模。开发模型的重要一环是机器学习中的“特征工程”。
  - pandas与其它分析库通常是靠NumPy的数组联系起来的。将DataFrame转换为NumPy数组，可以使用.values属性。
  
## 用Patsy创建模型描述

  - Patsy是Python的一个库，使用简短的字符串“公式语法”描述统计模型（尤其是线性模型）。Patsy适合描述statsmodels的线性模型。
  - patsy.dmatrices函数接收一个公式字符串和一个数据集（可以是DataFrame或数组的字典），为线性模型创建设计矩阵。
  - 这些Patsy的DesignMatrix实例是NumPy的ndarray，带有附加元数据。
  
## statsmodels介绍

  - statsmodels是Python进行拟合多种统计模型、进行统计试验和数据探索可视化的库。statsmodels包含许多经典的统计方法，但没有贝叶斯方法和机器学习模型。
  - statsmodels包含的模型有：
    - 线性模型，广义线性模型和健壮线性模型
    - 线性混合效应模型
    - 方差（ANOVA）方法分析
    - 时间序列过程和状态空间模型
    - 广义矩估计
  - 估计线性模型:
    - statsmodels的线性模型有两种不同的接口：基于数组和基于公式。它们可以通过API模块引入。
      ```
      import statsmodels.api as sm
      import statsmodels.formula.api as smf
      ```
    - statsmodels有多种线性回归模型，包括从基本（比如普通最小二乘）到复杂（比如迭代加权最小二乘法）的。
  - 估计时间序列过程：
    - statsmodels的另一模型类是进行时间序列分析，包括自回归过程、卡尔曼滤波和其它态空间模型，和多元自回归模型。
  
## scikit-learn介绍

  - scikit-learn是一个广泛使用、用途多样的Python机器学习库。它包含多种标准监督和非监督机器学习方法和模型选择和评估、数据转换、数据加载和模型持久化工具。这些模型可以用于分类、聚合、预测和其它任务。
  
    
