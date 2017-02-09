function [z, llh] = hmmViterbi_(M, A, s)
% Implmentation function of Viterbi algorithm. 
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 starting probability (prior)
% Output:
%   z: 1 x n latent state
%   llh:  loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
[k,n] = size(M);
Z = zeros(k,n);
A = log(A);
M = log(M);
Z(:,1) = 1:k;
v = log(s(:))+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A,v),[],1);    % 13.68
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[llh,idx] = max(v);
z = Z(idx,:);

