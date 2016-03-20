%% Collapse Gibbs sampling for Dirichelt process gaussian mixture model
close all; clear;
d = 2;
k = 3;
n = 500;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);

[y,model] = mixGaussGb(X);
figure
plotClass(X,y);