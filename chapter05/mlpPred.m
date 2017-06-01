function Y = mlpPred(model, X)
% Multilayer perceptron prediction
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
L = length(W)+1;
Y = X;
for l = 2:L
    Y = sigmoid(W{l-1}'*Y);
end