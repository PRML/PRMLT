function y = mlpPred(model, X)
% Multilayer perceptron prediction
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
L = length(W)+1;
Z = cell(L);
Z{1} = X;
for l = 2:L
    Z{l} = sigmoid(W{l-1}'*Z{l-1});
end
y = Z{L};
