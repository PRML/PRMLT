% demos for ch14
%% Mixture of linear regression
close all; clear
d = 1;
k = 2;
n = 500;
[X,y] = mixLinRnd(d,k,n);
plot(X,y,'.');
[label,model,llh] = mixLinReg(X, y, k);
plotClass([X;y],label);
figure
plot(llh);
[y_,z,p] = mixLinPred(model,X,y);
figure;
plotClass([X;y],label);

%% Mixture of logistic regression
d = 2;
c = 2;
k = 4;
n = 500;
[X,t] = kmeansRnd(d,c,n);

model = mixLogitBin(X,t-1,k);
y = mixLogitBinPred(model,X);
plotClass(X,y+1)
%% adaboost
d = 2;
k = 2;
n = 500;
[X,t] = kmeansRnd(d,k,n);
model = adaboostBin(X,t-1);
y = adaboostBinPred(model,X);
plotClass(X,y+1)

