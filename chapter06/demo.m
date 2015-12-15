clear; close all;

n = 100;
x = linspace(0,2*pi,n);   % test data
t = sin(x)+rand(1,n)/2;

% model = knReg(x,t,1e-4,@knGauss);
% y = knRegPred(model, x);
% figure;
% hold on;
% plot(x,t,'o');
% plot(x,y,'r-');


%% test case for kernel regression
lambda = 1e-4;
model_kn = knReg(x,t,lambda,@knLin);
model_lin = linReg(x,t,lambda);

[y_kn, s_kn] = knRegPred(model_kn, x);
[y_lin, s_lin] = linPred(model_lin,x);

maxabsdiff(y_kn,y_lin)
maxabsdiff(s_kn,s_lin)
%% test case for knCenter
% kn = @knGauss;
% X=rand(2,100);
% X1=rand(2,10);
% X2=rand(2,5);
% 
% isequalf(knCenter(kn,X,X1),diag(knCenter(kn,X,X1,X1)))
% isequalf(knCenter(kn,X),knCenter(kn,X,X,X));