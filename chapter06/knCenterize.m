function Kc = knCenterize(kn, X, Xt)
% Centerize the data in the kernel space
K = kn(X,X);
mK = mean(K);
mmK = mean(mK);
if nargin < 3
    % Kc = K-M*K-K*M+M*K*M; where M = ones(n,n)/n; 
    Kc = K+mmK-bsxfun(@plus,mK,mK');
else
    Kt = kn(X,Xt);
    mKt = mean(Kt);
    Kc = Kt+mmK-bsxfun(@plus,mKt,mK');
end