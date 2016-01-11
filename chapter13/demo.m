% demo

d = 3;
k = 2;
n = 10000;

[ X, M, A, E, s ] = hmmRnd(d, k, n);
[z, v] = hmmViterbi(M, A, s);

% [model, energy] = hmmEm(x,k);
% [alpha,energy] = hmmFwd(M,A,s);
% beta = hmmBwd(M,A);
% gamma = normalize(alpha.*beta,1);

