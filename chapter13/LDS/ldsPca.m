function [A, C, Z] = ldsPca(X, k, m)
% Subspace method for learning linear dynamic system.
% Input:
%   X: d x n data matrix
%   k: dimension of hidden variable
%   m: stacking order for the Hankel matrix
% Output:
%   A: k x k transition matrix
%   C: k x d emission matrix
%   Z: k x n latent variable
%   Y: d x n reconstructed data
% reference: Bayesian Reasoning and Machine Learning (BRML) chapter 24.5.3 p.507
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
H = reshape(X(:,hankel(1:m,m:n)),d*m,[]);
[U,S,V] = svd(H,'econ');
C = U(1:d,1:k);
Z = S(1:k,1:k)*V(:,1:k)';
A = Z(:,2:end)/Z(:,1:end-1); % estimated transition
% Y = C*Z; % reconstructions