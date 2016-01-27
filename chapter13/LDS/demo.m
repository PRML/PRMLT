% demo
clear; close all;
d = 3;
k = 2;
n = 2000;
% 
[X,model] = ldsRnd(d,k,n);
%% WARNING: The standard kalman filter as descripted in PRML is numerically unstable. Sometime, this demo fails, that is not my implementation problem.
[mu, V, llh] = kalmanFilter(X, model);
% % [nu, U, Ezz, Ezy, llh] = kalmanSmoother(X, model);
% [model, llh] = ldsEm(X, model);
% plot(llh);

