function [y, P] = logitMnPred(model, X)
% Prodict the label for multiclass (multinomial) logistic regression model
%   model: trained model structure
%   X: d x n testing data
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
X = [X; ones(1,size(X,2))];
A = W'*X;                                   
P = exp(bsxfun(@minus,A,logsumexp(A,1)));  
[~, y] = max(P,[],1);