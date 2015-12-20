clear; close all;
n = 100;
x = linspace(0,2*pi,n);   % test data
t = sin(x)+rand(1,n)/2;

model = knReg(x,t,1e-4,@knGauss);
knRegPlot(model,x,t);

%% kernel regression with linear kernel is linear regression
% clear; close all;
% n = 100;
% x = linspace(0,2*pi,n);   % test data
% t = sin(x)+rand(1,n)/2;
% lambda = 1e-4;
% model_kn = knReg(x,t,lambda,@knLin);
% model_lin = linReg(x,t,lambda);
% 
% idx = 1:2:n;
% xt = x(:,idx);
% tt = t(idx);
% 
% [y_kn, sigma_kn,p_kn] = knRegPred(model_kn,xt,tt);
% [y_lin, sigma_lin,p_lin] = linPred(model_lin,xt,tt);
% 
% maxabsdiff(y_kn,y_lin)
% maxabsdiff(sigma_kn,sigma_lin)
% maxabsdiff(p_kn,p_lin)
%% kernel kmeans with linear kernel is kmeans
% clear; close all;
% d = 2;
% k = 3;
% n = 500;
% [X,y] = kmeansRnd(d,k,n);
% init = ceil(k*rand(1,n));
% [y_kn,en_kn,model_kn] = knKmeans(X,init,@knLin);
% [y_lin,en_lin,model_lin] = kmeans(X,init);
% 
% idx = 1:2:n;
% Xt = X(:,idx);
% 
% [t_kn,ent_kn] = knKmeansPred(model_kn, Xt);
% [t_lin,ent_lin] = kmeansPred(model_lin, Xt);
% 
% maxabsdiff(y_kn,y_lin)
% maxabsdiff(en_kn,en_lin)
% 
% maxabsdiff(t_kn,t_lin)
% maxabsdiff(ent_kn,ent_lin)
%% kernel PCA with linear kernel is PCA
clear; close all;
d = 10;
p = 2;
n = 500;
X = randn(d,n);

model_lin = pca(X,p);
model_kn = knPca(X,p,@knLin);

idx = 1:2:n;
Xt = X(:,idx);
Y_lin = pcaPred(model_lin,Xt);
Y_kn = knPcaPred(model_kn,Xt);

R = Y_lin/Y_kn    % the results are equivalent up to a rotation.

%% test case for knCenter
% clear; close all;
% kn = @knGauss;
% X=rand(2,100);
% X1=rand(2,10);
% X2=rand(2,5);
% 
% isequalf(knCenter(kn,X,X1),diag(knCenter(kn,X,X1,X1)))
% isequalf(knCenter(kn,X),knCenter(kn,X,X,X));