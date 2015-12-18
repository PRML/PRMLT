% Done
% demo for chapter 03
clear; close all;
d = 1;
n = 200;
[x,t] = linRnd(d,n);
%%
model = linReg(x,t);
linPlot(model,x,t);
fprintf('Press any key to continue. \n');
%%
[model,llh] = linRegEbEm(x,t);
linPlot(model,x,t);
figure;
plot(llh);
fprintf('Press any key to continue. \n');

%%
[model,llh] = linRegEbFp(x,t);
[y, sigma] = linPred(model,x,t);
linPlot(model,x,t);
figure;
plot(llh);