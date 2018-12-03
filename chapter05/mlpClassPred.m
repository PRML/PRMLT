function [y, P] = mlpClassPred(model, X)
% Multilayer perceptron classification prediction
% logistic activation function is used.
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   y: 1 x n label vector
%   P: k x n probability matrix
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
b = model.b;
T = length(W);
Z = X;
for t = 1:T-1
    Z = sigmoid(W{t}'*Z+b{t});
end
P = softmax(W{T}'*Z+b{T});
[~,y] = max(P,[],1);  