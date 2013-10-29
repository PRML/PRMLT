function x = rndDirichlet(a)
% Sampling from a Dirichlet distribution.
% Written by Mo Chen (sth4nth@gmail.com).
x = gamrnd(a,1);
x = x/sum(x);