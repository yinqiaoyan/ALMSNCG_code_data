function [w,s,flag_cvg,Info_out] = almsncg_SVR(y,B,B_T,flag_prnt,Info_in)
% This code is based on the ALM-SNCG algorithm in
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
% yanyinqiao@ruc.edu.cn.%% function [w,s,flag_cvg,Info_out] = almsncg_SVC(B,B_T,flag_admm,flag_prnt,Info_in)
% Input: 
%          y -- Labels of training data
%               Each element of y takes value of 1 or -1.
%          B -- =-y.*X
%               X is the covariate matrix of training data. 
%               Each raw of X is a sample, with extra column of 
%               all ones added to the end of X.
%        B_T -- Transpose of B
%  flag_prnt -- =1 Print information of the algorithm
%               =0 Do not print information
%    Info_in -- Input parameters
%          Info_in.C - Penalty parameter
%          Info_in.sigma0 - Starting point of sigma
%          Info_in.tau - Transformation rate of sigma
%          Info_in.ep  - epsilon in eps-L1-loss SVR
%
%
% Output:
%          w -- slope and intercept
%          s -- =Bw - y
%   flag_cvg -- =1 The algorithm converges eventually
%               =0 Not converge               
%   Info_out -- Total iteration number of each method
%          Info_out.alm = k+1
%          Info_out.sn  = sniterwhole
%          Info_out.cg  = cgiterwhole
%          Info_out.linsch = lineiterwhole

%%
% Our code is designed to solve the eps-L1-loss SVR problem:
%     min_{w,s} 0.5 * \|w\|^2 + p_{eps}(s)
%     s.t.      s = B * w - y
% where p_{eps}(s) = C * \sum_{i=1}^{m} max(|s_i| - eps, 0).
%
% In augmented Lagrangian method (ALM), Lagrange multipliers are updated by
%     lambda^{k+1} = lambda^{k} - sigma_{k} * (s^{k+1} - B * w^{k+1} + y)
%     sigma_{k+1}  = tau * sigma_{k}
%%


 %% Initialization
[l,n] = size(B);
%%% Semismooth Newton-CG %%%
maxitSN   = 50;      % Maximum number of iteration
epsilonSN = 1.0e-2;
maxitPCG  = 200;
tolPCG    = 1.0e-6;

%%% Augmented Lagrangian method %%%
maxitALM = 10;
sigma_max = 2.0;
theta = 0.8;
tmp0 = 1.0e-6; 
lambda0 = zeros(l,1);
w0 = ones(n,1);      % Starting point of w
flag_cvg = 0;

%%% Three significant parameters in this algorithm %%%
C = 1.0;
sigma0 = 0.15;
tau = 1.0;
ep = 0.01;
if isfield(Info_in, 'C');  C = Info_in.C; end 
if isfield(Info_in, 'sigma0');  sigma0 = Info_in.sigma0; end 
if isfield(Info_in, 'tau');  tau = Info_in.tau; end 
if isfield(Info_in, 'ep');  ep = Info_in.ep; end 
fprintf('\n--------------------Start Training by Augmented Lagrangian Method---------------\n\n')
%% Augmented Lagrangian Method
w = w0;
sigma_k = sigma0;
lambda_k = lambda0;

jsn = 1;
cgiterwhole = 0;
sniterwhole = 0;
lineiterwhole = 0;
stmp4 = 0;

for k = 0:maxitALM 
    stmp4_0 = stmp4;
    Csigma_k = C/sigma_k;

    epsilonSN2 = max(epsilonSN, 1.0 * 10^(-jsn));
    if k == 0
        epsilonSN2 = 1;
    end
    
    [w,s,~,iterk_k,tmpSN, resSN,cgiterall,norm_F,line_iterall] = Semismooth_Newton(...
        w,B,B_T,lambda_k,sigma_k,C,Csigma_k,maxitSN,epsilonSN2,maxitPCG,tolPCG,y,ep,flag_prnt,n ); 

    tmp2 = tmpSN'*tmpSN;
    idx = find(s>0);
    cc = zeros(l,1); cc(idx)=C;
    tmp3 = tmpSN*sigma_k+cc-lambda_k;
    tmp4 = tmp3'*tmp3 + norm_F^2;
    stmp4 = sqrt(tmp4);
    
    if flag_prnt == 1
    fprintf('k=%d,tmpSN=%f, semi-iter_k=%d, epsilonSN=%e,tolCG=%e, resSN=%f, total-cgiter = %d, sigma_k= %f\n',...
        k,tmp2,iterk_k,epsilonSN2,tolPCG,resSN,cgiterall,sigma_k)
    end

    epsilonk = 1/(k+1);
    tolalm = epsilonk/max(1,sqrt(sigma_k));
    
    if flag_prnt == 1
    fprintf('tolalm=%f, sqrt_tmp4=%f, tmp3=%f,sum_s=%d\n',...
        tolalm,stmp4,norm(tmp3),sum(s>0))
    end
    
    if (stmp4<tolalm || abs(stmp4-stmp4_0)<0.01)
        flag_cvg = 1;
        cgiterwhole = cgiterwhole+cgiterall;
        sniterwhole = sniterwhole+iterk_k;
        lineiterwhole = lineiterwhole + line_iterall;
        break;
    end
    lambda_k = lambda_k - sigma_k * tmpSN;
    if (sigma_k < sigma_max) && (tmp2/tmp0 > theta)
        sigma_k = tau * sigma_k;
    end
    tmp0 = tmp2;
    
    % Update 
    jsn = jsn + 1;
    cgiterwhole = cgiterwhole + cgiterall;
    sniterwhole = sniterwhole + iterk_k;
    lineiterwhole = lineiterwhole + line_iterall;
    
end

Info_out.alm = k + 1;
Info_out.sn  = sniterwhole;
Info_out.cg  = cgiterwhole;
Info_out.linsch = lineiterwhole;

if flag_prnt == 1
fprintf('Total alm-iter=%d, Total cg-iter=%d, Total semi-iter=%d\n',...
    k+1,cgiterwhole,sniterwhole)
end
fprintf('\n--------------------End of Training by Augmented Lagrangian Method---------------\n\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% end of the main program %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Semismooth Newton method %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Semismooth Newton method for solving subproblem S1: grad_phi(w)=0
function [w,s,flag,iterk,tmpSN,resSN,cgiterall,norm_F,line_iterall]=Semismooth_Newton(...
    w0,B,B_T,lambda,sigma,C,Csigma,maxitSN,epsilonSN,maxitPCG,tolPCG,y,ep,flag_prnt,n )
%%% Initialization
eta = 0.5;  % eta in (0,+infty)
%   line search
mu   = 1.0e-4; % mu is in (0,1/2)
rho  = 0.5;  % rho is in (0,1)
%   Constants used later
ylambda = -y + lambda/sigma;
lamlam = lambda'*lambda/(2*sigma); 

w = w0;
flag = 0; % =1 if the algorithm converges

Bw = B_T'*w;

z = Bw + ylambda;
s = s_argmin(Csigma,z,ep);
phi = PhiValue( w,lamlam,sigma,C,z,s,ep );

iterk_cgall = 0;
line_iterall = 0;

for j = 1:maxitSN

    [F,tmpSN] = FValue( w,B,lambda,sigma,s,Bw,y );

    norm_F = sqrt(F'*F);
    
    if (norm_F <= epsilonSN)
        iterk = j;
        resSN = norm_F;
        cgiterall = iterk_cgall;
        flag = 1;
        break;
    end
    if j == maxitSN
        iterk = j;
        resSN = norm_F;
        cgiterall = iterk_cgall;
        fprintf('Semi-Newton does not converge\n')
        break;
    end
    b = -F;
    
    %%% pre_cg
    [d_j,~,~,iterk_cg,tolreal] = pre_cg(...
        b,maxitPCG,tolPCG,B_T,sigma,z,Csigma,n,norm_F,ep,eta );
    iterk_cgall = iterk_cgall + iterk_cg;

    phi0 = phi;
    [w,s,phi,z,Bw,k] = line_search(...
        w,d_j,rho,mu,phi0,F,lamlam,sigma,C,Csigma,B_T,ylambda,ep,Bw );
    line_iterall = line_iterall+k;
    
    if flag_prnt == 1
    fprintf('j=%d, norm_F = %f, cg-tolreal = %f, iterk_cg = %d, iter_line_search = %d\n',...
        j,norm_F,tolreal,iterk_cg,k)
    end

end
return
%%% End of Semismooth_Newton.m





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PCG method %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is exactly the algorithm given by Hestenes and Stiefel (1952)
%% An iterative method to solve A(x) =b  
%% The symmetric positive definite matrix M is a preconditioner for A.
%% See Pages 527 and 534 of Golub and va Loan (1996)

function [p,flag,relres,k1,tol] = pre_cg( b,maxit,tol,B_T,sigma,z,Csigma,n,norm_F,ep,eta )

idx = find( ( z>ep & z<(ep+Csigma) ) | ( z>(-ep-Csigma) & z<(-ep) ) );
B_new_T = B_T(:,idx);
p = zeros(n,1);
prec = 1;  % Precondition. Default is prec=1.
r = b;

if norm_F <= 1.0e-12
    k1  = 0;
    relres = 0;         
    flag   = 0;
    return;
end

if norm_F > 1.0e2
    maxit = 1;
end 

tol = min(eta, 0.05 * norm_F^2);  % tolerance
flag = 0;

%%% preconditioning 
z_cg   = r./prec;   
rz1 = r'*z_cg; 
rz2 = 1; 
d_cg   = z_cg;

%%% CG iteration
len_idx = length(idx);
P = 50;
if (len_idx<P) && (norm_F < 1.0e2) && (n<500)
    BnewTBnew = B_new_T*B_new_T';  % Accelerate the computation.

    for k1 = 1:maxit
        if k1 > 1
           beta = rz1/rz2;
           d_cg = z_cg + beta*d_cg;
        end
        tmp1 = (d_cg' * BnewTBnew')';
        tmp2 = sigma * tmp1;
        Ax = d_cg + tmp2;

        denom = d_cg'*Ax;

        if denom <= 0 
           p = d_cg/norm(d_cg);  % d is not a descent direction
           fprintf('denom < 0')
           break;  % exit
        else
           alpha = rz1/denom;
           p = p + alpha*d_cg;
           r = r - alpha*Ax;
        end
        z_cg = r./prec;
        norm_r = sqrt(r'*r);
        relres = norm_r / norm_F;  % relative residue = norm(r) / norm(b)
        if norm_r <= tol          
           flag = 1;
           break
        end
        rz2 = rz1;
        rz1 = r'*z_cg;
    end
else
    for k1 = 1:maxit
        if k1 > 1
           beta = rz1/rz2;
           d_cg = z_cg + beta*d_cg;
        end

        tmp1 = ((B_new_T'*d_cg)'*B_new_T')';
        tmp2 = sigma * tmp1;
        Ax = d_cg + tmp2;

        denom = d_cg'*Ax;

        if denom <= 0 
           p = d_cg/norm(d_cg);           
           fprintf('denom < 0')
           break;  % exit
        else
           alpha = rz1/denom;
           p = p + alpha*d_cg;
           r = r - alpha*Ax;
        end
        z_cg = r./prec;
        norm_r = sqrt(r'*r);
        relres = norm_r/norm_F; 
        if norm_r <= tol
           flag = 1;
           break
        end
        rz2 = rz1;
        rz1 = r'*z_cg;
    end
end

return
%%% End of pre_cg.m




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Line search %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w_new,s_new,phi,z_new,Bw,k,l] = line_search(...
    w0,dj,rho,mu,phi0,F0,lamlam,sigma,C,Csigma,B_T,ylambda,ep,Bw0 )

w_new = w0 + dj; % w_new = w0 + rho^m * d_j
Bdj = B_T'*dj;
Bw = Bw0 + Bdj;

z_new = Bw + ylambda;
s_new = s_argmin(Csigma,z_new,ep);
phi = PhiValue( w_new,lamlam,sigma,C,z_new,s_new,ep );
l = F0'*dj;
l = mu*l;

k = 0;
while  (phi-phi0 > l+1.0e-4) && (k<10)  % Slight relaxation of Armijo
    dj = dj * rho;
    w_new = w0 + dj;
    
    Bdj = Bdj * rho;
    Bw = Bw0 + Bdj;
    z_new = Bw + ylambda;
    s_new = s_argmin(Csigma,z_new,ep);
    phi = PhiValue( w_new,lamlam,sigma,C,z_new,s_new,ep );
    l = l*rho;
    k = k+1;
end
return
%%% End of line_search.m








%%% To get the value of function F(w), i.e. grad_phi(w)
function [F,tmp1] = FValue( w,B,lambda,sigma,s,Bw,y )

tmp1 = s - Bw + y;
tmp2 = lambda - sigma*tmp1;
tmp = B'*tmp2;
F = w + tmp;
return
%%% End of FValue.m



%%% To get the value of function phi(w)
function phi = PhiValue( w,lamlam,sigma,C,z,s,ep )

p = C*sum(max(abs(s)-ep,0));
tmp = s-z;
tmpz = tmp'*tmp;
w2 = w'*w;
phi = 0.5*sigma*tmpz + p + 0.5*w2 - lamlam; 
return
%%% End of PhiValue.m



%%% To get the value of s_argmin(w)
function [s] = s_argmin( Csigma,z,ep )
s = max( min( z , max(z-Csigma,ep) ) , min(z+Csigma,-ep) );
return
%%% End of s_argmin.m


