function [model, L] = mlpReg(X, y, k, lambda)
% Train a multilayer perceptron neural network for regression with backpropagation
% tanh activation function is used
% Input:
%   X: d x n data matrix
%   y: 1 x n real value response vector
%   k: T x 1 vector to specify number of hidden nodes in each layer
%   lambda: regularization parameter
% Ouput:
%   model: model structure
%   L: (regularized least square) loss
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    lambda = 1e-2;
end
eta = 1e-5;
tol = 1e-5;
maxiter = 50000;
L = inf(1,maxiter);

k = [size(X,1);k(:);size(y,1)];
T = numel(k)-1;
W = cell(T,1);
b = cell(T,1);
for t = 1:T
    W{t} = randn(k(t),k(t+1));
    b{t} = randn(k(t+1),1);
end
R = cell(T,1);
Z = cell(T+1,1);
Z{1} = X;
for iter = 2:maxiter
%     forward
    for t = 1:T-1
        Z{t+1} = tanh(W{t}'*Z{t}+b{t});             % 5.10 5.113
    end
    Z{T+1} = W{T}'*Z{T}+b{T};                       % 5.114

%     loss
    E = Z{T+1}-y;     
    Wn = cellfun(@(x) dot(x(:),x(:)),W);            % |W|^2
    L(iter) = dot(E(:),E(:))+lambda*sum(Wn);
    if abs(L(iter)-L(iter-1)) < tol*L(iter-1); break; end
    
%     backward
    R{T} = E;                
    for t = T-1:-1:1
        df = 1-Z{t+1}.^2;    % h'(a)
        R{t} = df.*(W{t+1}*R{t+1});    % 5.66
    end
    
%     gradient descent
    for t=1:T
        dW = Z{t}*R{t}'+lambda*W{t};    % 5.67
        db = sum(R{t},2);
        W{t} = W{t}-eta*dW;             % 5.43
        b{t} = b{t}-eta*db;
    end
end
L = L(2:iter);
model.W = W;
model.b = b;
