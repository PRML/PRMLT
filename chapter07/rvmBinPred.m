function [y, p] = rvmBinPred(model, X)
% Prodict the label for binary logistic regression model
%   model: trained model structure
%   X: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
index = model.index;
X = [X;ones(1,size(X,2))];
X = X(index,:);
w = model.w;
p = exp(-log1pexp(w'*X)); 
y = (p>0.5)+0;
