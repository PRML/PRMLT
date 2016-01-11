% TODO: 
%   1) beta for em regress
%   2) refine kmeansRnd and mixGaussRnd
%% demo: kmeans 
% close all; clear;
% d = 2;
% k = 3;
% n = 500;
% [X,label] = kmeansRnd(d,k,n);
% y = kmeans(X,k);
% plotClass(X,label);
% figure;
% plotClass(X,y);

%% demo: Em for Gauss Mixture 
close all; clear;
d = 2;
k = 3;
n = 1000;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);

m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
% train
[z1,model,llh] = mixGaussEm(X1,k);
figure;
plot(llh);
figure;
plotClass(X1,z1);
% predict
z2 = mixGaussPred(X2,model);
figure;
plotClass(X2,z2);
%% demo: Em for Gauss mixture initialized with kmeans;
% close all; clear;
% d = 2;
% k = 3;
% n = 500;
% [X,label] = mixGaussRnd(d,k,n);
% init = kmeans(X,k);
% [z,model,llh] = mixGaussEm(X,init);
% plotClass(X,label);
% figure;
% plotClass(X,init);
% figure;
% plotClass(X,z);
% figure;
% plot(llh);
