clear all; close all
%% Test for mixLinReg
% d = 1;
% k = 3;
% n = 500;
% W = randn(d+1,k);
% 
% [x, label] = rndKmeans(d, k, n);
% X = [x; ones(1,n)];
% y = zeros(1,n);
% for j = 1:k
%     idx = (label == j);
%     y(idx) = W(:,j)'*X(:,idx);
% end
% 
% plot(x,y,'.');
% [model, label,llh] = mixLinReg(X, y, 3);
% spread([x;y],label);
% figure
% plot(llh);

%%
[X, y] = rndKmeans(2,3,1000);
[label,L] = mixGaussVb(X, 10);
plot(L);