function [model, energy] = regressVb(X, t, prior)
% Fit empirical Bayesian linear model with EM
% X: m x n data
% t: 1 x n response
if nargin < 3
    a0 = 1e-4;
    b0 = 1e-4;
    c0 = 1e-4;
    d0 = 1e-4;
else
    a0 = prior.a;
    b0 = prior.b;
    c0 = prior.c;
    d0 = prior.d;
end
[m,n] = size(X);

xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

XX = X*X';
Xt = X*t';

maxiter = 100;
energy = -inf(1,maxiter+1);
dg = sub2ind([m,m],1:m,1:m);
I = eye(m);
tol = 1e-8;

a = a0+m/2;
c = c0+n/2;
Ealpha = 1e-4;
Ebeta = 1e-4;
for iter = 2:maxiter
    invS = Ebeta*XX;
    invS(dg) = invS(dg)+Ealpha;
    U = chol(invS);
    Ew = Ebeta*(U\(U'\Xt));
    
    w2 = dot(Ew,Ew);
    e2 = sum((t-Ew'*X).^2);    
    invU = U\I;   
    trS = dot(invU(:),invU(:));
    invUX = U\X;
    trXSX = dot(invUX(:),invUX(:));
    
    b = b0+0.5*(w2+trS);
    d = d0+0.5*(e2+trXSX);
    
    Ealpha = a/b;
    Ebeta = c/d; 
        
    logdetS = -2*sum(log(diag(U)));        
    energy(iter) = -a*log(b)-c*log(d)+0.5*logdetS;
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end
end
const = gammaln(a)-gammaln(a0)+gammaln(c)-gammaln(c0)+a0*log(b0)+c0*log(d0)+0.5*(m-n*log(2*pi));
energy = energy(2:iter)+const;
w0 = tbar-dot(Ew,xbar);

model.w0 = w0;
model.w = Ew;
model.Ealpha = Ealpha;
model.Ebeta = Ebeta;
model.a = a;
model.b = b;
model.c = c;
model.d = d;
model.xbar = xbar;
