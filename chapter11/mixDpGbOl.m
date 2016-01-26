function [label, Theta, w, llh] = mixDpGbOl(X, alpha, theta)
% Online collapsed Gibbs sampling for Dirichlet process (infinite) mixture model (a.k.a.
% DPGM). Any component model can be used, such as Gaussian
n = size(X,2);
Theta = {};
nk = [];
label = zeros(1,n);
for i = randperm(n)
    x = X(:,i);
    Pk = log(nk)+cellfun(@(t) t.logPredPdf(x), Theta);
    P0 = log(alpha)+theta.logPredPdf(x);
    p = [Pk,P0];
    k = discreteRnd(exp(p-logsumexp(p)));
    if k == numel(Theta)+1
        Theta{k} = theta.clone().addSample(x);
        nk = [nk,1];
    else
        Theta{k} = Theta{k}.addSample(x);
        nk(k) = nk(k)+1;
    end
    label(i) = k;
end
w = nk/n;