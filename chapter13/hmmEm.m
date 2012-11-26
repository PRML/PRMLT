function [model, energy] = hmmEm(x, k, A, E, s)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% x: 1 x n sequence of observations
% A: k x k transition matrix
% E: k x m emission matrix
% s: k x 1 initial probabilisty
n = size(x,2);
d = max(x);
X = sparse(x,1:n,1,d,n);

if nargin < 3    
    A = normalize(rand(k,k),2);
    E = normalize(rand(k,d),2);
    s = normalize(rand(k,1),1);
end
M = E*X;

tol = 1e-4;
maxIter = 100;
energy = -inf(1,maxIter);
for iter = 2:maxIter
%     E-step
    [alpha,energy(iter)] = hmmFwd(M,A,s);
    beta = hmmBwd(M,A);
    Gamma = normalize(alpha.*beta,1);
    Xi = A.*(alpha(:,1:n-1)*(beta(:,2:n).*M(:,2:n))');
    
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end   % check likelihood for convergence
%     M step 
    s = Gamma(:,1);
    A = normalize(Xi,2);    
    E = bsxfun(@times,Gamma*X',1./sum(Gamma,2));
    M = E*X;
end
energy = energy(2:iter);
model.A = A;
model.E = E;
model.s = s;


