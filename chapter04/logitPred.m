function [y, p] = logitPred(model, X)
% Prodict the label for binary lgoistic regression model
% model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing label
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
p = exp(-log1pexp(w'*X+w0)); 
y = (p>0.5)+0;

