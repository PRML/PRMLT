function [gamma, alpha, beta, c] = hmmSmoother_(M, A, s)
% Implmentation function HMM smoothing alogrithm.
% Unlike the method described in the book of PRML, the alpha and beta
% returned is the normalized.
% Computing unnormalized version alpha and beta is numerical unstable, which grows exponential fast to infinity.
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 start prior probability
% Output:
%   gamma: k x n matrix of posterior gamma(t)=p(z_t,x_{1:T})
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:T})
%   beta: k x n matrix of posterior beta(t)=gamma(t)/alpha(t)
%   c: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
[K,T] = size(M);
At = A';
c = zeros(1,T); % normalization constant
alpha = zeros(K,T);
[alpha(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),c(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);  % 13.59
end
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = A*(beta(:,t+1).*M(:,t+1))/c(t+1);   % 13.62
end
gamma = alpha.*beta;                  % 13.64

