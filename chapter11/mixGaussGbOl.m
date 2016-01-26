function [label,Theta,nk] = mixGaussGbOl(X,kappa0,m0,nu0,S0,alpha0)
[d,n] = size(X);
mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
s = sum(Xo(:).^2)/(d*n);
if nargin == 1
    kappa0 = 1;
    m0 = mean(X,2);
    nu0 = d;
    S0 = s*eye(d);
    alpha0 = 1;
end
Theta = {};
nk = [];
prior = GaussWishart(kappa0,m0,nu0,S0);
label = zeros(1,n);
for i = randperm(n)
    K = numel(Theta);
    x = X(:,i);
    Pk = nk.*exp(cellfun(@(theta) theta.logPredPdf(x),Theta));
    P0 = alpha0*exp(prior.logPredPdf(x));
    k = discreteRnd(normalize([Pk,P0]));
    if k==K+1
        Theta{k} = GaussWishart(kappa0,m0,nu0,S0).addSample(x);
        nk = [nk,1];
    else
        Theta{k} = Theta{k}.addSample(x);
        nk(k) = nk(k)+1;
    end
    label(i) = k;
end