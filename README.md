# ALMSNCG_code_data
Code and Data for "An efficient augmented Lagrangian method for support vector machine"

## Implement ALM-SNCG

### Code

* almsncg_SVC.m
* almsncg_SVR.m
* TestSVC.m
* TestSVR.m

### Datasets

* abalone.mat
* w2a.mat

## Example

### Support vector machine for classification (SVC）

```matlab
TestSVC
% ==== The Information of Dataset ====
% Dataset: w2a
% Total samples and features: (46279, 300)
% Nonzeros: 539213
% Density: 3.88%
% Proportion of training data: 80%
% Size of training and testing data: (37023, 9256)
% ============================
% ==== Finish Training! ====
% Convergence state: 1
% Training time: 0.2319
% Accuracy: 99.9676%
```

### Support vector machine for regression (SVR）

```matlab
TestSVR
% ==== The Information of Dataset ====
% Dataset: abalone
% Total samples and features: (4177, 8)
% Nonzeros: 32080
% Density: 96.00%
% Proportion of training data: 60%
% Size of training and testing data: (2506, 1671)
% ============================
% ==== Finish Training! ====
% Convergence state: 1
% Training time: 0.0035
% MSE: 12.8510
```
