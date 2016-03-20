% demos for LDS in ch13

clear; close all;
d = 3;
k = 2;
n = 100;
 
[X,Z,model] = ldsRnd(d,k,n);
[mu, V, llh] = kalmanFilter(X, model);

[nu, U, Ezz, Ezy, llh] = kalmanSmoother(X, model);
[model, llh] = ldsEm(X, model);
plot(llh);

