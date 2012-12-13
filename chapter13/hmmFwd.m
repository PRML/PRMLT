function [alpha, energy] = hmmFwd(M, A, s)


[K,T] = size(M);
At = A';
energy = zeros(1,T);
alpha = zeros(K,T);
[alpha(:,1),energy(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),energy(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);
end
energy = sum(log(energy(energy>0)));