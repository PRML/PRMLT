% clear; close all;
% 

%% sparse signal recovery demo
clear; close all; 

d = 512; % signal length
k = 20;  % number of spikes
n = 100; % number of measurements
%
% random +/- 1 signal
x = zeros(d,1);
q = randperm(d);
x(q(1:k)) = sign(randn(k,1)); 

% projection matrix
A = unitize(randn(d,n),1);
% noisy observations
sigma = 0.005;
e = sigma*randn(1,n);
y = x'*A + e;
[model,llh] = rvmRegEbSeq(A,y);
plot(llh);


% [model,llh] = rvmRegEbFp(A,y);
% plot(llh);
m = zeros(d,1);
m(model.index) = model.w;

h = max(abs(x))+0.2;
x_range = [1,d];
y_range = [-h,+h];
figure;
subplot(2,1,1);plot(x); axis([x_range,y_range]); title('Original Signal');
subplot(2,1,2);plot(m); axis([x_range,y_range]); title('Recovery Signal');
% 
% [y, sigma] = rvmRegPred(model,A);
%% regression
% d = 100;
% beta = 1e-1;
% X = rand(1,d);
% w = randn;
% b = randn;
% t = w'*X+b+beta*randn(1,d);
% 
% x = linspace(min(X)-1,max(X)+1,d);   % test data
%%
% [model,llh] = rvmRegFp(X,t);
% figure
% plot(llh);
% [y, sigma] = linInfer(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off
%%
% [model,llh] = rvmRegEm(X,t);
% figure
% plot(llh);
% [y, sigma] = linInfer(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off
%%
% [model,llh] = rvmRegSeq(X,t);
% figure
% plot(llh);
% [y, sigma] = linPred(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off

%% classification
% clear; close all
% k = 2;
% d = 2;
% n = 1000;
% [X,t] = kmeansRnd(d,k,n);
% [x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
% 
% [model, llh] = rvmBinFp(X,t-1);
% plot(llh);
% y = rvmBinPred(model,X)+1;
% figure;
% binPlot(model,X,y);

