function [ gamma, c ] = hmmRecSmoother_( M, A, s )
% Forward-backward (recursive gamma no alpha-beta) alogrithm for HMM to compute posterior p(z_i|x)
% Input:
%   x: 1xn observation
%   s: kx1 starting probability of p(z_1|s)
%   A: kxk transition probability
%   E: kxd emission probability
% Output:
%   gamma: 1xn posterier p(z_i|x)
%   llh: loglikelihood or evidence lnp(x)
% Written by Mo Chen sth4nth@gmail.com
[K,T] = size(M);
At = A';
c = zeros(1,T); % normalization constant
gamma = zeros(K,T);
[gamma(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [gamma(:,t),c(t)] = normalize((At*gamma(:,t-1)).*M(:,t),1);  % 13.59
end
for t = T-1:-1:1
    gamma(:,t) = normalize(bsxfun(@times,A,gamma(:,t)),1)*gamma(:,t+1);
end

