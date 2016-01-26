% TODO: multiPlot: plot multiclass decison boundary
% 
clear; close all;
k = 2;
n = 1000;
[X,t] = kmeansRnd(2,k,n);
[model, llh] = logitBin(X,t-1,0);
plot(llh);
y = logitBinPred(model,X)+1;
binPlot(model,X,y)
pause
%%
% clear
% k = 3;
% n = 1000;
% [X,t] = kmeansRnd(2,k,n);
% [model, llh] = logitMn(X,t);
% y = logitMnPred(model,X);
% plotClass(X,y)
