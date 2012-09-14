function y = pdflogVmfLn(X, mu, kappa)
% Compute log pdf of a von Mises-Fisher distribution.
% Written by Mo Chen (mochen80@gmail.com).
d = size(X,1);
c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
q = bsxfun(@times,mu,kappa)'*X;
y = bsxfun(@plus,q,c');
