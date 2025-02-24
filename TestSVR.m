function TestSVR
%%%%%%%%%%%%%%%
%%% Test of SVR
% In the paper, we use 60% of the dataset as training set,
% 40% as testing set.
%
% Parameter Settings:
%   C = 1 / n_tr * 5;
%   sigma0 = 0.1;
%   tau = 1.0;
%   ep = 0.01;


% This code is based on the ALM-SNCG algorithm in
% ``An Efficient Augmented Lagrangian Method for Support Vector Machine''
% By Yinqiao Yan* and Qingna Li**
% %Optimization Methods and Software, 2020，DOI: 10.1080/10556788.2020.1734002
% * yanyinqiao@ruc.edu.cn
% ** qnl@bit.edu.cn,

% Received date: 17 Apr. 2019
% Accepted date: 12 Feb. 2020
% 
% Last updated: 27 Aug. 2020
% If you have any problems, please contact us by qnl@bit.edu.cn or
% yanyinqiao@ruc.edu.cn.


%% Load the dataset
load('abalone.mat')
[n,m] = size(Xdata);
Xdata = full(Xdata);
nonzeros = sum(sum(Xdata ~= 0));
density = nonzeros / (n * m) * 100;

% Proportion of training set
training_proportion = 0.6;

n_tr = floor(n * training_proportion);
n_te = n - n_tr;
Xtrain = Xdata(1:n_tr,:);
Ytrain = Ydata(1:n_tr);
Xtest  = Xdata(n_tr+1:end,:);
Ytest  = Ydata(n_tr+1:end);

fprintf('==== The Information of Dataset ====\n')
fprintf('Dataset: %s\n', char('abalone'))
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
C = 1 / n_tr * 5;
sigma0 = 0.1;
tau = 1.0;
ep  = 0.01;

Info_in.C = C;
Info_in.sigma0 = sigma0;
Info_in.tau = tau;
Info_in.ep = ep;

%% Training
t0 = tic;
[w,s,flag,Info_out] = almsncg_SVR(Ytrain, B, B_T, flag_prnt, Info_in);
t1 = toc(t0);
fprintf('==== Finish Training! ====\n')
fprintf('Convergence state: %d\n', flag)
fprintf('Training time: %.4f\n', t1)

%% Testing
b = w(end);
wr = w(1:end-1);

Ymodel = Xtest * wr + b;
mse = sum((Ytest-Ymodel).^2)/n_te;

fprintf('MSE: %.4f\n', mse)

