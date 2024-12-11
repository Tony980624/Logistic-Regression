# 逻辑回归原理

$P_{label=1} = \frac{1}{1+e^{-(\beta_0+\beta_1x)}}$ , 它额输出值永远除以(0,1)区间，吻合了概率的特性，当线性方程部分的输出越大，函数就越接近1.

## 损失函数

$L = \sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)] , p_i 是模型预测为1的概率$ ， 目标是最大化这个值。

所以逻辑回归的损失函数和线性回归截然不同，它的参数也不是一次就得来的，需要迭代

常用于逻辑回归的优化方法有：牛顿法

## 牛顿法原理详解

$\beta^(t+1)=\beta^t-H^{-1}\nabla f(\beta^t)$

$\beta^t 是当前参数值，\nabla f(\beta^t)是目标函数的一阶导数，H：目标函数f(\beta)的海森(Hessian Matrix)矩阵（二阶导数矩阵，描述函数的曲率）$

为了更好的理解这个公式，现在降维到1维

泰勒series,函数在a点时的值可以近似为: $f(x) \approx f(a)+f'(a)(x-a)+\frac{1}{2}f''(a)(x-a)^2$

对这个系列，对X求导，得到 $f'(a)+f''(a)(x-a)=0$

求解得到 $x = a-\frac{f'(a)}{f''(a)}, 这也就是牛顿优化的更新公式x_{k+1} = x_k-\frac{f'(x_k)}{f''(x_k)} $

这个公式会自动朝着极值点移动，因为根据上面证明，它本身就是求导后设右边为0，也就是极值点。

同样的道理， $\beta^(t+1)=\beta^t-H^{-1}\nabla f(\beta^t)$ 这个公式其实和一维公式是一样的，只是把二阶导数在 $x_k$ 点的值替换成了Hessian矩阵，也就是多个变量的二阶导的矩阵，把一阶导数的值换成了向量。

## 信用卡违约数据案例

```
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv(f'D:/Download/UCI_Credit_Card.csv')
data.head()
# data.isna().sum()
Y = data.iloc[:,-1]
X = data.iloc[:,1:-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=72)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# 打印混淆矩阵
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("混淆矩阵:")
print(conf_matrix)
```
模型预测结果：

| 实际\预测 | 预测: 0 | 预测: 1 |
|-----------|----------|----------|
| 实际: 0   | 5661     | 218      |
| 实际: 1   | 1224     | 397      |   

对于这个案例，关注点应该是预测到了多少比例的违约人，这个模型只预测到了 397/(1224+397) ,比较低的准确率，远低于投色子。

## Lasso约束，正则的公式和灵感

上面讲到逻辑回归的损失函数是最大化这个 $L = \sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)]$ ，前面加一个负号，也就是变成最小化 $L = -\sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)] $.再加上一个惩罚项 $\lambda \sum_{j=1}^P |\beta_j|$, 损失函数变为 $L = -\sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)] + \lambda \sum_{j=1}^P |\beta_j|$, 要使它最小，说明惩罚项不能太大，要使不重要的系数为0才行。

## Lasso 规范化的原理和迭代方法

Lasso Logistic 回归使用的优化方法是梯度下降或其变种（如坐标下降法）。

优化目标： $L = -\sum^N_{i=1}[y_ilog(p_i)+(1-y_i)log(1-p_i)] + \lambda \sum_{j=1}^P |\beta_j|$

参数更新法: $\beta_j^{t+1} = \beta_j^t - \eta (\frac{\partial{L}}{\partial{\beta_j}}+sign(\beta_j)) $ 

$\eta$ 是学习率，步长

$$
\text{sign}(\beta_j) =
\begin{cases} 
+1, & \beta_j > 0 \\
-1, & \beta_j < 0 \\
0, & \beta_j = 0
\end{cases}
$$

符号函数，表面优化的方向。


