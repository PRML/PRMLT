% demos for ch03
clear; close all;
d = 1;
n = 200;
[x,t] = linRnd(d,n);
%% Linear regression
model = linReg(x,t);
plotBar(model,x,t);
%% Empirical Bayesian linear regression via EM
[model1,llh] = linRegEm(x,t);
plot(llh);
plotBar(model,x,t);
%%  Empirical Bayesian linear regression via Mackay fix point iteration method
[model,llh] = linRegFp(x,t);
[y, sigma] = linPred(model,x,t);
plot(llh);
plotBar(model,x,t);
