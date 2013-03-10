function [nu, U, UU, llh] = kalmanSmoother(X, model)
% DONE
% Kalman smoother
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu; % prior mean
P0 = model.P;  % prior covairance

n = size(X,2);
q = size(mu0,1);
mu = zeros(q,n);
V = zeros(q,q,n);
P = zeros(q,q,n); % C_{t+1|t}
Amu = zeros(q,n); % u_{t+1|t}
llh = zeros(1,n);
I = eye(q);

% forward
PC = P0*C';
R = (C*PC+S);
K = PC/R;
mu(:,1) = mu0+K*(X(:,1)-C*mu0);
V(:,:,1) = (I-K*C)*P0;
P(:,:,1) = P0;  % useless, just make a point
Amu(:,1) = mu0; % useless, just make a point
llh(1) = pdfGaussLn(X(:,1),C*mu0,R);
for i = 2:n    
    [mu(:,i), V(:,:,i), Amu(:,i), P(:,:,i), llh(i)] = ...
        forwardStep(X(:,i), mu(:,i-1), V(:,:,i-1), A, G, C, S, I);
end
llh = sum(llh);
% backward
nu = zeros(q,n);
U = zeros(q,q,n);
UU = zeros(q,q,n-1);
nu(:,n) = mu(:,n);
U(:,:,n) = V(:,:,n);
for i = n-1:-1:1  
    [nu(:,i), U(:,:,i), UU(:,:,i)] = ...
        backwardStep(nu(:,i+1), U(:,:,i+1), mu(:,i), V(:,:,i), Amu(:,i+1), P(:,:,i+1), A);
end

function [mu, V, Amu, P, llh] = forwardStep(x, mu, V, A, G, C, S, I)
P = A*V*A'+G;
PC = P*C';
R = C*PC+S;
K = PC/R;
Amu = A*mu;
CAmu = C*Amu;
mu = Amu+K*(x-CAmu);
V = (I-K*C)*P;
llh = pdfGaussLn(x,CAmu,R);

function [nu, U, UU] = backwardStep(nu, U, mu, V, Amu, P, A)
J = V*A'/P;   % smoother gain matrix
nu = mu+J*(nu-Amu);
UU = J*U; % Bishop eqn 13.104
U = V+J*(U-P)*J';
