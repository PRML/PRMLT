% demo for chapter 06
% TODO: 1) plot error bar for kernel regression
% 2) Derive proper formula for GP with kernel space centrization
% 3) GP for classification
clear; close all;
n = 100;
beta = 1e-1;
X = rand(1,n);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,n);

x = linspace(min(X)-1,max(X)+1,n);   % test data
%%
model = regressKn(X,t,1e-4,@knGauss);
y = knInfer(x,model);
figure;
hold on;
% plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off
% figure
% plot(llh);
axis equal