close all; clear;
d = 2;
k = 3;
n = 5000;
[X,label] = kmeansRnd(d,k,n);
y = litekmeans(X,k);
plotClass(X,label);
figure;
plotClass(X,y);
