% TODO: 
%   1) demo
%   2) pred
%   3) unify model parameter
% demo
d = 10;
m = 2;
n = 1000;

[X] = ppcaRnd(m,d,n);
%%
[model,llh] = pcaVb(X);
[model, llh] = pcaEm(X,m);
[model, llh] = fa(X,m);
plot(energy)