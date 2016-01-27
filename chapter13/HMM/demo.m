% demo

d = 3;
k = 2;
n = 1000;

[x, model] = hmmRnd(d, k, n);
[z, v] = hmmViterbi(x,model);
[alpha,energy] = hmmFilter(x,model);
[gamma, alpha, beta, c] = hmmSmoother(x,model);
[model, energy] = hmmEm(x,k);
plot(energy)
