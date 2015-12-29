% Done
% demo for chapter 03
clear; close all;
d = 1000;
n = 2000;
[x,t] = linRnd(d,n);
%%
% model = linReg(x,t);
% linPlot(model,x,t);
% fprintf('Press any key to continue. \n');
%%
[model1,llh1] = linRegEbEm(x,t);
% linPlot(model,x,t);
% figure;
% plot(llh);
% fprintf('Press any key to continue. \n');

%%
[model2,llh2] = linRegEbFp(x,t);
% [y, sigma] = linPred(model,x,t);
% % linPlot(model,x,t);
% figure;
% plot(llh);