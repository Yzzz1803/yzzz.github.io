---
tags: statistic
---

## Bias & Variance
### 1.  Bias
一般在统计学中，我们就是以样本的统计量来对母体进行估计，而Bias便是**度量我们样本统计量与母体的差异程度**。
- Estimate the mean of a variable x
	- assume the mean of x is $μ$
	- assume rhe variance of x is $\sigma^2$
- Estimator of variance $\sigma^2$
	- sample N points: { $x^1, x^2, ..., x^N$ }

$$
m=\frac{1}{N} \sum_n x^n \\
\quad s^2=\frac{1}{N} \sum_n\left(x^n-m\right)^2
$$

Biased estimator

$$
E[s^2] = \frac{N-1}{N} \sigma^2 \neq \sigma^2
$$

以上式为例，正常我们取样的样本平均值$m$很大的概率不会等于母体平均值，如果我们对母体做了很多次的抽样，其期望值$E(m) = \mu$，则我们称这样的抽样是*unbiased*的。

### 2. Variance
Variance(方差;变异数)在统计学中是一个度量样本分散程度的统计量
$$s^2=\frac{1}{N} \sum_n\left(x^n-m\right)^2$$

![](https://files.mdnice.com/user/43031/71145b7a-5aee-45fc-a640-be2afadedb6c.png)



当我们取样的数量越多，分散程度便会缩小，可以更接近我们的目标。从机器学习的角度来看，当我们的模型选择越复杂，整个Model Set(Hypothesis Set)会有越多种各式各样的选择，其Variance也会越高。
总结来说，我们可以用箭靶来描述Bias & Variance

![](https://files.mdnice.com/user/43031/4399a3cb-7213-40c4-aed5-ac2d63e68c6a.png)


## 3.Model-Error-Bias-Variance

### 3.1 越复杂的Model, Variance越高

![](https://files.mdnice.com/user/43031/126b4392-0865-42be-99fa-b93842e9ff28.png)


### 3.2 越复杂的Model, Bias越低

![](https://files.mdnice.com/user/43031/d7de0aa1-bcb1-43e9-81b7-9bbd8e20f844.png)


但是当我们选择越复杂的模型，整个Variance产生的Error上升的速度远高于Bias产生的error下降的速度，这是我们选择越复杂的模型时，整个test error会飙的非常高的原因。


![](https://files.mdnice.com/user/43031/370572b8-5391-4bac-84ec-edd7a8f03909.png)


由此我们也可以给underfitting与overfitting一个比较清楚的定义：

***Underfitting*** : Large bias and Small variance \
***Overfitting*** : Small bias and Large variance

