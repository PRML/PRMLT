d = 2;
n = 100;
X = randn(d,n);
x = rand(d,1);
mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
Sigma = Xo*Xo'/n;

XX = Xo*Xo'/n;
XXX = X*X'/n-mu*mu';
U = chol(XX);

U_ = chol(Xo(:,2:end)*Xo(:,2:end)');
UU = cholupdate(U,(X(:,1)-mu));




p = logGauss(x,mu,Sigma);
gauss = Gaussian(X(:,3:end));
gauss = gauss.addSample(X(:,1));
gauss = gauss.addSample(X(:,2));
p2 = gauss.logPdf(x);