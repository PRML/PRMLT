% TODO:
% 1) sparse visualization
% 2) sparse data/demos
% 3) fix coordinate descent
% 4) sparse prediction for regression and classification

% 5) need to be extensively tested on high dimensional data (inlucding
% chapter03 Bayesian linear regression)



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


[model,llh] = rvmRegEbEm(A,y);
plot(llh);

% solve by BCS
tic;
[weights,used,sigma2,errbars] = BCS_fast_rvm(A,y,initsigma2,1e-8);
t_BCS = toc;
fprintf(1,'BCS number of nonzero weights: %d\d',length(used));
x_BCS = zeros(d,1); err = zeros(d,1);
x_BCS(used) = weights; err(used) = errbars;


E_BCS = norm(x-x_BCS)/norm(x);

figure
subplot(3,1,1); plot(x); axis([1 d -max(abs(x))-0.2 max(abs(x))+0.2]); title(['(a) Original Signal']);
subplot(3,1,3); errorbar(x_BCS,err); axis([1 d -max(abs(x))-0.2 max(abs(x))+0.2]); title(['(c) Reconstruction with BCS, n=' num2str(n)]); box on;

disp(['BCS: ||I_hat-I||/||I|| = ' num2str(E_BCS) ', time = ' num2str(t_BCS) ' secs']);
%% regression
% d = 100;
% beta = 1e-1;
% X = rand(1,d);
% w = randn;
% b = randn;
% t = w'*X+b+beta*randn(1,d);

% x = linspace(min(X)-1,max(X)+1,d);   % test data
%%
% [model,llh] = rvmRegEbFp(X,t);
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
% [model,llh] = rvmRegEbEm(X,t);
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
% [model,llh] = rvmRegEbCd(X,t);
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
% n = 2;
% d = 2;
% d = 1000;
% [X,t] = rndKCluster(d,n,d);
% [x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),d), linspace(min(X(2,:)),max(X(2,:)),d));

%%
% [model, llh] = rvmEbFp(X,t-1);
% figure
% plot(llh);
% figure;
% spread(X,t);
% 
% w = zeros(3,1);
% w(model.used) = model.w;
% y = w(1)*x1+w(2)*x2+w(3);
% hold on;
% contour(x1,x2,y,1);
% hold off;
%%
% [model, llh] = rvmEbEm(X,t-1);
% figure
% plot(llh);
% figure;
% spread(X,t);
% 
% w = model.w;
% y = w(1)*x1+w(2)*x2+w(3);
% hold on;
% contour(x1,x2,y,1);
% hold off;