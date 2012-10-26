clear; close all;

%% regression
n = 100;
beta = 1e-1;
X = rand(1,n);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,n);

x = linspace(min(X)-1,max(X)+1,n);   % test data
%%
[model,energy] = regressVb(X,t);
% figure
plot(energy);
y = linInfer(x,model);
figure;
hold on;
% plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off
