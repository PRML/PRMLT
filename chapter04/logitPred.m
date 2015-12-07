function [y, p] = logitPred(model, X)
% Prodict the label for binary logistic regression model
%   model: trained model structure
%   X: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
p = exp(-log1pexp(w'*X+w0)); 
y = (p>0.5)+0;

