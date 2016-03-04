function [y, R] = nbGaussPred(model, X)


mu = model.mu;
sigma = model.sigma;
a = model.a;


lambda = 1./sigma;
ml = mu.*lambda;
q = bsxfun(@plus,X'.^2*lambda-2*X'*ml,dot(mu,ml,1)); % M distance
c = d*log(2*pi)+2*sum(log(sigma),1); % normalization constant
R = -0.5*bsxfun(@plus,q,c);
y = max(R,[],1);