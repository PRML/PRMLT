function [A, nodePot, edgePot] = mrfIsGa(im, sigma, J)
% Contruct a latent Ising MRF with Gaussian observation
% Input:
%   im: row x col image
%   sigma: variance of Gaussian node potential
%   J: parameter of Ising edge
% Output:
%   A: n x n adjacent matrix
%   nodePot: 2 x n node potential
%   edgePot: 2 x 2 x m edge potential
% Written by Mo Chen (sth4nth@gmail.com)
A = lattice(size(im));
[s,t,e] = find(triu(A));
m = numel(e);
e(:) = 1:m;
A = sparse([s;t],[t;s],[e;e]);

z = [1;-1];
x = reshape(im,1,[]);
nodePot = -(x-z).^2/(2*sigma^2);
edgePot = repmat(J*(z*z'),[1, 1, m]);