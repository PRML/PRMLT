% demo for ch08

%% Naive Bayes with Gauss
d = 2;
k = 3;
n = 1000;
[X, t] = kmeansRnd(d,k,n);
plotClass(X,t);

model = nbGauss(X,t);