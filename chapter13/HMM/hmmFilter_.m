function [alpha, energy] = hmmFilter_(M, A, s)
% HMM forward filtering algorithm
% Unlike the method described in the book of PRML, the alpha returned is the normalized version: alpha(t)=p(z_t|x_{1:t})
% The unnormalized version is numerical unstable. alpha(t)=p(z_t,x_{1:t}) grows exponential fast to infinity.
% Written by Mo Chen (sth4nth@gmail.com).
[K,T] = size(M);
At = A';
energy = zeros(1,T);
alpha = zeros(K,T);
[alpha(:,1),energy(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),energy(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);
end
energy = sum(log(energy(energy>0)));