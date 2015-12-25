% 
clear; close all;
k = 2;
n = 1000;
[X,t] = kmeansRnd(2,k,n);
[model, llh] = logitBin(X,t-1,0);
plot(llh);
binPlot(model,X,t)
pause
%%
clear
k = 3;
n = 1000;
[X,t] = kmeansRnd(2,k,n);
[model, llh] = logitMn(X,t);
y = logitMnPred(model,X);
spread(X,y)
