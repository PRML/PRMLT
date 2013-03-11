function [alpha, energy] = hmmFilter(M, A, s)
% HMM forward filtering algorithm
% unlike the method described in the book of PRML
% the alpha returned is the normalized version: alpha(t)=p(z_t|x_{1:t})
% the unnormalized version alpha(t)=p(z_t,x_{1:t}) grows exponential fast
% to infinity.
[K,T] = size(M);
At = A';
energy = zeros(1,T);
alpha = zeros(K,T);
[alpha(:,1),energy(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),energy(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);
end
energy = sum(log(energy(energy>0)));