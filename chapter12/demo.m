% demo
d = 10;
m = 2;
n = 1000;

[X] = ppcaRnd(m,d,n);
%%
[model,energy] = pcaVb(X);
[model, llh] = pcaEm(X,3);
[model, llh] = fa(X,3);
plot(energy)