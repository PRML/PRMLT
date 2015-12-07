function [y, p] = logitPred(model, X)
% Prodict the label for binary lgoistic regression model
% model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing label
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
a = w'*X+w0;
y = a > 0;
h = ones(1,size(X,2));
h(~y) = -1;
p = exp(-sum(log1pexp(-h.*a))); 

% (p > 0.5)==y
