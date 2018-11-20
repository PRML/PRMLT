function [model, L] = mlpReg(X,Y,k,lambda)
% Train a multilayer perceptron neural network
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   k: T x 1 vector to specify number of hidden nodes in each layer
%   lambda: regularization parameter
% Ouput:
%   model: model structure
%   L: loss
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    lambda = 1e-2;
end
eta = 1e-3;
maxiter = 50000;
L = inf(1,maxiter);

k = [size(X,1);k(:);size(Y,1)];
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
        Z{t+1} = tanh(W{t}'*Z{t}+b{t});
    end
    Z{T+1} = W{T}'*Z{T}+b{T};

%     loss
    E = Z{T+1}-Y;     
    Wn = cellfun(@(x) dot(x(:),x(:)),W);            % |W|^2
    L(iter) = dot(E(:),E(:))+lambda*sum(Wn);

%     backward
    R{T} = E;                % delta
    for t = T-1:-1:1
        df = 1-Z{t+1}.^2;    % h'(a)
        R{t} = df.*(W{t+1}*R{t+1});    % delta
    end
    
%     gradient descent
    for t=1:T
        dW = Z{t}*R{t}'+lambda*W{t};
        db = sum(R{t},2);
        W{t} = W{t}-eta*dW;
        b{t} = b{t}-eta*db;
    end
end
L = L(1,2:iter);
model.W = W;
model.b = b;
