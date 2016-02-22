% TODO: 
% 1) test against matlab implementation of kalman filter
% 2) simplify ldsEm with less parameters (G=diag(g), S=I) 

% demos for LDS in ch13

clear; close all;
d = 3;
k = 2;
n = 100;
 
[X,Z,model] = ldsRnd(d,k,n);
plot(Z(1,:),Z(2,:),'-');
plot(X(1,:),X(2,:),'-');
[mu, V, llh] = kalmanFilter(X, model);

[nu, U, Ezz, Ezy, llh] = kalmanSmoother(X, model);
[model, llh] = ldsEm(X, model);
plot(llh);

