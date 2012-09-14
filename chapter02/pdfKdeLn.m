function z = pdfKdeLn (X, Y, sigma)
% Compute log pdf of kernel density estimator.
% Written by Mo Chen (mochen80@gmail.com).
    D = bsxfun(@plus,full(dot(X,X,1)),full(dot(Y,Y,1))')-full(2*(Y'*X));
    z = logSumExp(D/(-2*sigma^2),1)-0.5*log(2*pi)-log(sigma*size(Y,2));
endfunction
