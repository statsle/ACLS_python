# ACLS_python
Adaptive Capped Least Squares (python package)
## Description
This package includes the randomized gradient descent method method applied to minimize the adaptive capped least squares loss.

Suppose we observe data vectors  (x<sub>i</sub>,y<sub>i</sub>) that follow a linear model y<sub>i</sub>=x<sub>i</sub><sup>T</sup>&beta;<sup>*</sup>+&epsilon;<sub>i</sub>, i=1,...n, where y<sub>i</sub> is a univariate response,  x<sub>i</sub> is a d-dimensional predictor, &beta;<sup>*</sup> denotes the vector of regression coefficients, and &epsilon;<sub>i</sub> is a random error. We propose the adpative capped least squares loss, &ell;(x)=x<sup>2</sup>/2 if |x| &leq; &tau;; &tau;<sup>2</sup>/2, if |x| &gt; &tau;, where &tau;=&tau;(n) &gt; 0 is referred to as the adaptive capped least squares parameter. The proposed methods are applied to find &beta; that minimizes L(&beta;)= n<sup>-1</sup> &sum; &ell;(y<sub>i</sub>-x<sub>i</sub><sup>T</sup> &beta; ).

## Installation
Install **ACLS_python** from GitHub:
``` python
pip install git+https://github.com/rruimao/ACLS_python.git
from ACLS.RGD_bindings import RGD
``` 
