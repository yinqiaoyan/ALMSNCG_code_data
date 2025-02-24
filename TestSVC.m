function TestSVC
%%%%%%%%%%%%%%%% This code is based on the ALM-SNCG algorithm in
% ``An Efficient Augmented Lagrangian Method for Support Vector Machine''
% By Yinqiao Yan* and Qingna Li**
% 
% * yanyinqiao@ruc.edu.cn
% ** qnl@bit.edu.cn,
%Optimization Methods and Software, 2020，DOI: 10.1080/10556788.2020.1734002
% Received date: 17 Apr. 2019
% Accepted date: 12 Feb. 2020
% 
% Last updated: 27 Aug. 2020
% If you have any problems, please contact us by qnl@bit.edu.cn or
% yanyinqiao@ruc.edu.cn.

%%% Test of SVC
% In the paper, we use 80% of the dataset as training set,
% 20% as testing set.
%
% Parameter Settings:
%   C = 1 / n_tr * 550;
%   sigma0 = 0.15;
%   tau = 1.0;

%% Load the dataset
load('w2a.mat')
[n,m] = size(Xdata);
Xdata = full(Xdata);
nonzeros = sum(sum(Xdata ~= 0));
density = nonzeros / (n * m) * 100;

%% Split the dataset
training_proportion = 0.8;

n_tr = floor(n * training_proportion);
n_te = n - n_tr;
Xtrain = Xdata(1:n_tr,:);
Ytrain = Ydata(1:n_tr);
Xtest  = Xdata(n_tr+1:end,:);
Ytest  = Ydata(n_tr+1:end);

fprintf('==== The Information of Dataset ====\n')
fprintf('Dataset: %s\n', char('w2a'))
fprintf('Total samples and features: (%d, %d)\n', n, m)
fprintf('Nonzeros: %d\n', nonzeros)
fprintf('Density: %.2f%%\n', density)
fprintf('Proportion of training data: %d%%\n', training_proportion * 100)
fprintf('Size of training and testing data: (%d, %d)\n', n_tr, n_te)
fprintf('============================\n')

%% Reformulate the training set
XtrainALM = [Xtrain, ones(n_tr,1)];
B = XtrainALM;
ind = Ytrain == 1;
B(ind,:) = -B(ind,:);
B_T = B';

%% Parameters settings
flag_prnt = 0;
C = 1 / n_tr * 550;
sigma0 = 0.15;
tau = 1.0;

Info_in.C = C;
Info_in.sigma0 = sigma0;
Info_in.tau = tau;

%% Training
t0 = tic;
[w,s,flag,Info_out] = almsncg_SVC(B, B_T, flag_prnt, Info_in);
t1 = toc(t0);
fprintf('==== Finish Training! ====\n')
fprintf('Convergence state: %d\n', flag)
fprintf('Training time: %.4f\n', t1)

%% Testing
N = length(Ytest);
b = w(end);
wr = w(1:end-1);

Ymodel = Xtest * wr + b;
Ymodel = double(Ymodel>0);
Ymodel(Ymodel==0) = -1;

correct_num = sum(Ymodel == Ytest);
accuracy = correct_num / N * 100;
fprintf('Accuracy: %2.4f%%\n', accuracy)

