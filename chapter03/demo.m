% Done
% demo for chapter 03
clear; close all;
d = 1;
n = 200;
[x,t] = linRnd(d,n);
%%
% model = linReg(x,t);
% linPlot(model,x,t);
%%
% [model1,llh1] = linRegEm(x,t);
% plot(llh);
% linPlot(model,x,t);
%%
[model,llh] = linRegFp(x,t);
[y, sigma] = linPred(model,x,t);
plot(llh);
linPlot(model,x,t);
