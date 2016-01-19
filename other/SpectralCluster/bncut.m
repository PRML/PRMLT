function [y, ratio] = bncut(W)
% Bipartitioning normalized cut
mincut = 1;   % minimal number of nodes to be cut off

n = size(W,2);
[L,d] = laplacian(W,'n');
V = symeig(L,2)';
%%
[~,idx] = sort(V(2,:)./sqrt(d));
Vol_A = cumsum(d(idx));
Vol_B = sum(d)-Vol_A;

S = triu(W(idx,idx));
W_AB = full(cumsum(sum(S'-S,1)));

ratios = W_AB.*(1./Vol_A+1./Vol_B)/2;
[ratio,cut] = min(ratios(mincut:n-mincut));
y = true(1,n);
y(idx(1:cut+mincut-1)) = false;
