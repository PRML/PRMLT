function [X, label, model] = mixGaussRnd(d, k, n)
% Sampling form a Gaussian mixture distribution.
% Written by Michael Chen (sth4nth@gmail.com).
alpha0 = 1;  % hyperparameter of Dirichlet prior
W0 = eye(d);  % hyperparameter of inverse Wishart prior of covariances
v0 = d+1;  % hyperparameter of inverse Wishart prior of covariances
mu0 = zeros(d,1);  % hyperparameter of Guassian prior of means
beta0 = 1/(nthroot(k,d))^2; % hyperparameter of Guassian prior of means

w = dirichletRnd(ones(alpha0,k));
z = discreteRnd(w,n);

mu = zeros(d,k);
Sigma = zeros(d,d,k);
X = zeros(d,n);
for i = 1:k
    idc = z==i;
    Sigma(:,:,i) = iwishrnd(W0,v0); % invpd(wishrnd(W0,v0));
    mu(:,i) = gaussRnd(mu0,Sigma(:,:,i)/beta0);
    X(:,idc) = gaussRnd(mu(:,i),Sigma(:,:,i),sum(idc));
end
label = z;
model.mu = mu;
model.Sigma = Sigma;
model.weight = w;