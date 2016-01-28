clear; close all;
d = 3;
m = 2;
n = 1000;

X = ppcaRnd(m,d,n);
plotClass(X);
%% PCA , EM PCA and Constraint EM PCA produce the same result in the sense of reconstruction error
% classical PCA
[U,L,mu,err1] = pca(X,m);
Y = U'*bsxfun(@minus,X,mu);   % projection
Z1 = bsxfun(@times,Y,1./sqrt(L));  % whiten
figure;
plotClass(Y);
figure;
plotClass(Z1);
err1
% EM PCA
[W2,Z2,mu,err2] = pcaEm(X,m);
figure;
plotClass(Z1);
err2
% Contrained EM PCA
[W3,Z3,mu,err3] = pcaEmC(X,m);
figure;
plotClass(Z1);
err3
%% EM probabilistic PCA
[W,mu,beta,llh] = ppcaEm(X,m);
plot(llh)

%% Variational Bayesian probabilistic PCA
[model, energy] = ppcaVb(X);
plot(energy);

%% factor analysis
[W, mu, psi, llh] = fa(X, m);
plot(llh);
