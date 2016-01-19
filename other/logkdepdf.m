function r = logkdepdf(X, Y, sigma2)

d = size(X,1);
r = logsumexp(sqdistance(Y,X)/(-2*sigma2)-(log(2*pi)+d*log(sigma2))/2,1);