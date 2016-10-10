% demos for HMM in ch13

d = 3;
k = 2;
n = 10000;
[x, model] = hmmRnd(d, k, n);
%%
[z,p] = hmmViterbi(x,model);
%%
[alpha,llh] = hmmFilter(x,model);
%%
[gamma,alpha,beta,c] = hmmSmoother(x,model);
%%
[model, llh] = hmmEm(x,k);
plot(llh)
