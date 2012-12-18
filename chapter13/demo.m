% demo
% x = rand(10,1000);
% [model, energy] = dimPcaVb(x);
d = 3;
n = 100000;

k = 2;

A = normalize(rand(k,k),2);
E = normalize(rand(k,d),2);
s = normalize(rand(k,1),1);

z = zeros(1,n);
x = zeros(1,n);
z(1) = discreternd(s);
x(1) = discreternd(E(z(1),:));
for i = 2:n
    z(i) = discreternd(A(z(i-1),:));
    x(i) = discreternd(E(z(i),:));
end
X = sparse(x,1:n,1,d,n);
M = E*X;

[model, energy] = hmmEm(x,k);
% [alpha,energy] = hmmFwd(M,A,s);
% beta = hmmBwd(M,A);
% gamma = normalize(alpha.*beta,1);

% [gamma2,alpha2.beta2,loglik] = hmmFwdBack(s, A, M);