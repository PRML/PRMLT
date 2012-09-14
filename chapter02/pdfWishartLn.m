function y = pdfWishartLn(Sigma, v, W)
% Compute log pdf of a Wishart distribution.
% Written by Mo Chen (mochen80@gmail.com).
d = length(Sigma);
B = -0.5*v*logdet(W)-0.5*v*d*log(2)-logmvgamma(0.5*v,d);
y = B+0.5*(v-d-1)*logdet(Sigma)-0.5*trace(W\Sigma);