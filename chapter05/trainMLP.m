function [Wx,Wy,mse]=trainMLP(X,T,k)
% The matrix implementation of the Backpropagation algorithm for two-layer
% Multilayer Perceptron (MLP) neural networks.
% Input parameters:
%   X: Input matrix.  X is a (d x n) dimensional matrix, where d is a number of the inputs and n is a training size.
%   T: Desired response matrix. T is a (p x n) dimensional matrix, where p is a number of the output neurons and n is a training size.
%   k: Number of hidden neurons
% Output parameters:
%   Wx: Hidden layer weight matrix. Wx is a (k x d+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (p x k+1) dimensional matrix.
%   mse: Mean square error. 
[d,n] = size(X);
p = size(T,1);
eta = 1/n;

X = [ones(1,n) ; X];
Wx = rand(k,d+1);
Wy = rand(p,k+1);

maxiter = 500;
mse = zeros(1,maxiter);
for iter = 1:maxiter
%     forward
    Z = sigmoid(Wx*X);
    S = [ones(1,n);Z];
    Y = sigmoid(Wy*S);
    E = T-Y;
    mse(iter) = mean(dot(E(:),E(:)));
%     backward
    df = Y.*(1-Y);
    dGy = df.*E;
    dWy = dGy*S';
    Wy = Wy+eta*dWy;

    df = S.*(1-S);
    dGx = df.*(Wy'*dGy);
    dGx = dGx(2:end,:);
    dWx = dGx*X';
    Wx = Wx+eta*dWx;
    
end
mse = mse(1:iter);



