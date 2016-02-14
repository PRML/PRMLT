% TODO: 
% 1) test against matlab implementation of kalman filter
% 2) simplify ldsEm with less parameters (G=diag(g), S=I) 

clear; close all;
d = 3;
k = 2;
n = 6;
 
[X,model] = ldsRnd(d,k,n);
plot(X(1,:),X(2,:),'-');
[mu, V, llh] = kalmanFilter(X, model);

% [nu, U, Ezz, Ezy, llh] = kalmanSmoother(X, model);
% [model, llh] = ldsEm(X, model);
% plot(llh);

