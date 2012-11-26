function beta = hmmBwd(M, A)

[K,T] = size(M);
beta =  zeros(K,T);
beta(:,T) = 1;
for t = T-1:-1:1
    beta(:,t) = normalize(A*(beta(:,t+1).*M(:,t+1)),1);
end

