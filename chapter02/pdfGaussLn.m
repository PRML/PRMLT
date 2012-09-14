function y = pdfGaussLn(X, mu, sigma)
% Compute log pdf of a Gaussian distribution.
% Written by Mo Chen (mochen80@gmail.com).

[d,n] = size(X);
k = size(mu,2);
if n == k && size(sigma,1) == 1           
    X = bsxfun(@times,X-mu,1./sigma);
    q = dot(X,X,1);  % M distance
    c = d*log(2*pi)+2*log(sigma);          % normalization constant
    y = -0.5*(c+q);
elseif size(sigma,1)==d && size(sigma,2)==d && k==1   % one mu and one dxd sigma
    X = bsxfun(@minus,X,mu);
    [R,p]= chol(sigma);
    if p ~= 0
        error('ERROR: sigma is not PD.');
    end
    Q = R'\X;
    q = dot(Q,Q,1);  % quadratic term (M distance)
    c = d*log(2*pi)+2*sum(log(diag(R)));   % normalization constant
    y = -0.5*(c+q);
elseif size(sigma,1)==d && size(sigma,2)==k % k mu and k diagonal sigma
    lambda = 1./sigma;
    ml = mu.*lambda;
    q = bsxfun(@plus,X'.^2*lambda-2*X'*ml,dot(mu,ml,1)); % M distance
    c = d*log(2*pi)+2*sum(log(sigma),1); % normalization constant
    y = -0.5*bsxfun(@plus,q,c);
elseif size(sigma,1)==1 && (size(sigma,2)==k || size(sigma,2)==1) % k mu and (k or one) scalar sigma
    X2 = repmat(dot(X,X,1)',1,k);
    D = bsxfun(@plus,X2-2*X'*mu,dot(mu,mu,1));
    q = bsxfun(@times,D,1./sigma);  % M distance
    c = d*(log(2*pi)+2*log(sigma));          % normalization constant
    y = -0.5*bsxfun(@plus,q,c);
else
    error('Parameters mismatched.');
end
