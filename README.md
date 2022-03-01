# ACLS_python
Adaptive Capped Least Squares (python package)
## Description
This package includes the randomized gradient descent method method applied to minimize the adaptive capped least squares loss.

Suppose we observe data vectors  (x<sub>i</sub>,y<sub>i</sub>) that follow a linear model y<sub>i</sub>=x<sub>i</sub><sup>T</sup>&beta;<sup>*</sup>+&epsilon;<sub>i</sub>, i=1,...n, where y<sub>i</sub> is a univariate response,  x<sub>i</sub> is a d-dimensional predictor, &beta;<sup>*</sup> denotes the vector of regression coefficients, and &epsilon;<sub>i</sub> is a random error. We propose the adpative capped least squares loss, &ell;(x)=x<sup>2</sup>/2 if |x| &leq; &tau;; &tau;<sup>2</sup>/2, if |x| &gt; &tau;, where &tau;=&tau;(n) &gt; 0 is referred to as the adaptive capped least squares parameter. The proposed methods are applied to find &beta; that minimizes L(&beta;)= n<sup>-1</sup> &sum; &ell;(y<sub>i</sub>-x<sub>i</sub><sup>T</sup> &beta; ).

## Installation
Install **ACLS_python** from GitHub:
``` python
pip install git+https://github.com/rruimao/ACLS_python.git
``` 
## Function
- **RGD**: Randomized gradient descent method.


## Examples
We present two examples: random generated data with y-outliers and random generated data with x-outliers and y-outliers. 



### First example: random generated data with y-outliers
we generate contaminated random errors &epsilon;<sub>i</sub> from a mixture of normal distribution 0.9N(0,1)+0.1N(10,1) and x<sub>i</sub>'s are independently and identically distributed (i.i.d.) from N(0,&Sigma;) where &Sigma;=0.5<sup>|j-k|</sup>. We set &beta;<sup>*</sup> =(0,3,4,1,2,0)<sup>T</sup> to generate y<sub>i</sub>. We provide one example of this type, "ex_1.csv", and it can be downloaded from example file.

We randomly generate 10 initials &beta;<sup>*</sup> ~ Unif(B<sub>2</sub>(&tau;)), where Unif(B<sub>2</sub>(&tau;)) is a uniform distribution on the l<sub>2</sub>-ball B<sub>2</sub>(&tau;)={x: ||x||<sub>2</sub> &leq; &tau; }. This method finds the initial that provides the smallest adaptive capped least squares loss.

```python
from ACLS.RGD_bindings import RGD
import numpy as np
import pandas as pd
df = pd.read_csv('ex_1.csv')
Y=df['Y'].to_numpy()
X=df[['Intercept','X1','X2','X3','X4','X5']].to_numpy()
n=50
iter=10
eta_0=1e-3
alpha=2
tau=np.sqrt(n)/np.log(np.log(n))
beta_1=RGD(X,Y,tau,iter,eta_0,alpha)
```

### Second example: random generated data with x-outliers and y-outliers
we generate contaminated random errors &epsilon;<sub>i</sub> from a mixture of normal distribution 0.9N(0,1)+0.1N(10,1) and x<sub>i</sub>'s are independently and identically distributed (i.i.d.) from N(0,&Sigma;) where &Sigma;=0.5<sup>|j-k|</sup>. We then add a random perturbation vector  z<sub>i</sub> ~ N(10 &times; 1<sub>d-1</sub>,I<sub>d-1</sub>) to each covariate x<sub>i</sub> in the contaminated samples. We also use &beta;<sup>*</sup> =(0,3,4,1,2,0)<sup>T</sup> and use uncontaminated x<sub>i</sub> to generate y<sub>i</sub>. We provide one example of this type, "ex_2.csv", and it can be downloaded from example file.
	
``` python
df = pd.read_csv('ex_2.csv')
Y=df['Y'].to_numpy()
X=df[['Intercept','X1','X2','X3','X4','X5']].to_numpy()
n=50
iter=10
eta_0=1e-3
alpha=2
tau=np.sqrt(n)/np.log(np.log(n))
beta_2=RGD(X,Y,tau,iter,eta_0,alpha)
```

## Reference
Sun, Q., Mao, R. and Zhou, W.-X. Adaptive capped least squares. [Paper](https://arxiv.org/abs/2107.00109)
