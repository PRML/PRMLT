% Done
% demo for chapter 03
clear; close all;
n = 100;
beta = 1e-1;
X = rand(1,n);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,n);

x = linspace(min(X)-1,max(X)+1,n);   % test data
%%
model = regress(X, t);
y = linInfer(x, model);
figure;
hold on;
plot(X,t,'o');
plot(x,y,'r-');
hold off
%%
[model,llh] = regressEbEm(X,t);
[y, sigma] = linInfer(x,model,t);
figure;
hold on;
plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off
figure
plot(llh);
%%
[model,llh] = regressEbFp(X,t);
[y, sigma] = linInfer(x,model,t);
figure;
hold on;
plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off
figure
plot(llh);