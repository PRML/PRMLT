function [label, Theta, w, llh] = mixGaussGbOl(X, alpha, theta)
n = size(X,2);
Theta = {};
nk = [];
label = zeros(1,n);
for i = randperm(n)
    x = X(:,i);
    Pk = nk.*exp(cellfun(@(t) t.logPredPdf(x), Theta));
    P0 = alpha*exp(theta.logPredPdf(x));
    k = discreteRnd(normalize([Pk,P0]));
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