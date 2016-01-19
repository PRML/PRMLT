function label = discretize(V, d, m)
% Perform discretization on relaxed real value solution of spectral clustering
% V: k x n eigenvectors
% d: 1 x n degree vector
% Written by Michael Chen (sth4nth@gmail.com).
if nargin < 3
    m = 1;
end
switch m
    case 1
        label = ys(V,d);
    case 2
        label = njw(V);
    case 3
        label = bj(V,d);
    case 4
        label = zj(V(2:end,:),d);
    otherwise
        error('The parameter value of m is not supported.');
end

function label = ys(X, d) % Multiclass Spectral Clustering by S.Yu & J.Shi
[k,n] = size(X);
X = bsxfun(@times,X,1./sqrt(d+eps));
X = normalize(X);
idx = initialize(X);
R = X(:,idx);
% s = inf;
% while true
%     X = R'*X;
%     [~,label] = max(X,[],1);
%     [U,S,V] = svd(X*full(sparse(1:n,label,1,n,k,n))); 
%     
%     l = s;
%     s = trace(S);
%     if abs(s-l) < eps; break; end;
%     R = U*V';
% end
X = R'*X;
[~,label] = max(X,[],1);
last = 0;
while any(label ~= last)
    [U,~,V] = svd(X*full(sparse(1:n,label,1,n,k,n))); 
    R = U*V';
    X = R'*X;
    last = label;
    [~,label] = max(X,[],1);
end


function label = njw(X) % On Spectral Clustering by A.Y.Ng, M.I.Jordan & Y.Weiss
X = normalize(X);
idx = initialize(X);
label = wkmeans(X,idx,1); % standard kmeans.

function label = bj(X, d) % Learning Spectral Clustering by F.R.Bach & M.I.Jordans
X = bsxfun(@times,X,1./sqrt(d+eps));
idx = initialize(X);
label = wkmeans(X,idx,d);

function label = zj(X, d) % Multiway Spectral Clustering by Z.Zhang & M.I.Jordan
k = size(X,1)+1;
n = size(X,2);
G = eye(k,k-1)-repmat(1./k,k,k-1);
w = 1./sqrt(d+eps);
idx = initialize(X);
R = normalize(X(:,idx));
% s = inf;
% while true
%     Y = bsxfun(@times,R'*X,w);
%     [~,label] = max([Y;zeros(1,n)],[],1);
%     [U,S,V]=svd(X*full(sparse(1:n,label,1,n,k,n))*G);
% 
%     l = s;
%     s = trace(S);
%     if abs(s-l) < eps; break; end;
%     R = U*V';
% end
Y = bsxfun(@times,R'*X,w);
[~,label] = max([Y;zeros(1,n)],[],1);
last = 0;
while any(label ~= last)
    [U,~,V]=svd(X*full(sparse(1:n,label,1,n,k,n))*G);
    R = U*V';
    Y = bsxfun(@times,R'*X,w);
    last = label;
    [~,label] = max([Y;zeros(1,n)],[],1);
end

function idx = initialize(X)
% Choose k approximately orthogonal samples.
[k,n] = size(X);
X = normalize(X);
idx = zeros(1,k);
idx(1) = ceil(n*rand);
c = zeros(1,n);
for i = 2:k
    c = c+abs(X(:,idx(i-1))'*X);
    [~,idx(i)] = min(c);
end

function X = normalize(X)
% Normalize column vectors.
X = bsxfun(@times,X,1./sqrt(dot(X,X,1)));

function label = wkmeans(X, init, w)
% Perform weighted k-means initialized by centers.
[k,n] = size(X);
m = X(:,init);
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
last = 0;
while any(label ~= last)
    E = sparse(1:n,label,w,n,k,n);
    m = bsxfun(@times,X*E,1./full(sum(E,1)));
    last = label;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
end

