# Implement ALM-SNCG
# These codes are based on the paper:  
% ``An Efficient Augmented Lagrangian Method for Support Vector Machine''
% By Yinqiao Yan* and Qingna Li**
% Optimization Methods and Software, 2020，DOI: 10.1080/10556788.2020.1734002
% 
% * yanyinqiao@ruc.edu.cn
% ** qnl@bit.edu.cn
%
% Received date: 17 Apr. 2019
% Accepted date: 12 Feb. 2020
% 
% Last updated: 27 Aug. 2020
% If you have any problems, please contact us by qnl@bit.edu.cn or
% yanyinqiao@ruc.edu.cn.
## Code

 almsncg_SVC.m: main code for augmented Lagrangian Method for SVC
 almsncg_SVR.m: main code for augmented Lagrangian Method for SVR
 TestSVC.m: test code  for running almsncg_SVC, to call it, just type TestSVC in command window
 TestSVR.m: test code  for running almsncg_SVC, to call it, just type TestSVC in command window

## Datasets

 abalone.mat
 w2a.mat

# Example

## Support vector machine for classification (SVC）

```matlab
TestSVC
% ==== The Information of Dataset ====
% Dataset w2a
% Total samples and features (46279, 300)
% Nonzeros 539213
% Density 3.88%
% Proportion of training data 80%
% Size of training and testing data (37023, 9256)
% ============================
% ==== Finish Training! ====
% Convergence state 1
% Training time 0.2319
% Accuracy 99.9676%
```



## Support vector machine for regression (SVR）

```matlab
TestSVR
% ==== The Information of Dataset ====
% Dataset abalone
% Total samples and features (4177, 8)
% Nonzeros 32080
% Density 96.00%
% Proportion of training data 60%
% Size of training and testing data (2506, 1671)
% ============================
% ==== Finish Training! ====
% Convergence state 1
% Training time 0.0035
% MSE 12.8510
```

