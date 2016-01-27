function [gamma, alpha, beta, c] = hmmSmoother_(M, A, s)
% HMM smoothing alogrithm (normalized forward-backward or normalized alpha-beta algorithm)
% Written by Mo Chen (sth4nth@gmail.com).
[K,T] = size(M);
At = A';
c = zeros(1,T); % normalization constant
alpha = zeros(K,T);
[alpha(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),c(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);
end
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = A*(beta(:,t+1).*M(:,t+1))/c(t+1);
end
gamma = alpha.*beta;

