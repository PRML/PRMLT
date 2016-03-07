% demo for ch08

%% Naive Bayes with independent Gausssian
d = 2;
k = 3;
n = 1000;
[X, t] = kmeansRnd(d,k,n);
plotClass(X,t);

m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
t1 = t(1:m);
model = nbGauss(X1,t1);
y2 = nbGaussPred(model,X2);
plotClass(X2,y2);

%% Naive Bayes with independent Bernoulli
close all; clear;
d = 10;
k = 2;
n = 2000;
[X,t,mu] = mixBernRnd(d,k,n);
m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
t1 = t(1:m);
t2 = t((m+1):end);
model = nbBern(X1,t1);
y2 = nbBernPred(model,X2);
err = sum(t2~=y2)/numel(t2);