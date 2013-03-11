function [model, energy] = ldsEm(X, model)
% EM algorithm for parameter estimation of LDS
tol = 1e-4;
maxIter = 100;
energy = -inf(1,maxIter);
for iter = 2:maxIter
%     E-step
    [nu, U, Ezz, Ezy, energy(iter)] = kalmanSmoother(X, model);
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end   % check likelihood for convergence
%     M-step 
    model = mStep(X, nu, U, Ezz, Ezy);
end
energy = energy(2:iter);
model.A = A;
model.G = G;
model.C = C;
model.S = S;
model.mu0 = mu0;
model.P0 = P0;

function model = mStep(X ,nu, U, Ezz, Ezy)
n = size(X,2);
mu0 = nu(:,1);
P0 = U(:,:,1);

Ezzn = sum(Ezz,3);
Ezz1 = Ezzn-Ezz(:,:,n);
Ezz2 = Ezzn-Ezz(:,:,1);
Ezy = sum(Ezy,3);

A = Ezy/Ezz1;
EzyA = Ezy*A';
G = (Ezz2-(EzyA+EzyA')+A*Ezz1*A')/(n-1);
Xnu = X*nu';
C = Xnu/Ezzn;
XnuC = Xnu*C';
S = (X*X'-(XnuC+XnuC')+C*Ezzn*C')/n;

model.A = A;
model.G = G;
model.C = C;
model.S = S;
model.mu0 = mu0;
model.P0 = P0;
