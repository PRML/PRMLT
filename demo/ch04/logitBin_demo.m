% demos for ch04

%% Logistic logistic regression for binary classification
close all;
clear; 
d = 2;
k = 2;
n = 1000;
[X,t] = kmeansRnd(d,k,n);
[model, llh] = logitBin(X,t-1);
plot(llh);
y = logitBinPred(model,X)+1;
figure
binPlot(model,X,y)