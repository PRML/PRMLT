function Y=runMLP(X,Wx,Wy)
% The matrix implementation of the two-layer Multilayer Perceptron (MLP) neural networks.
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br
%
% Input parameters:
%   X: Input neural network.  X is a (p x K) dimensional matrix, where p is a number of the inputs and K >= 1.
%   Wx: Hidden layer weight matrix. Wx is a (H x p+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (m x H+1) dimensional matrix.
%
% Output parameters:
%  Y: Outpuy neural network.  Y is a (m x K) dimensional matrix, where m is a number of the output neurons and K >= 1.

[p1 N] = size (X);

bias = -1;

X = [bias*ones(1,N) ; X];

V = Wx*X;
Z = 1./(1+exp(-V));

S = [bias*ones(1,N);Z];
G = Wy*S;

Y = 1./(1+exp(-G));