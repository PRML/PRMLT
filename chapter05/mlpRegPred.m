function Y = mlpRegPred(model, X)
% Multilayer perceptron prediction
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
b = model.b;
T = length(W);
Z = cell(T+1,1);
Z{1} = X;
for t = 1:T-1
    Z{t+1} = tanh(W{t}'*Z{t}+b{t});
end
Y = W{T}'*Z{T}+b{T};