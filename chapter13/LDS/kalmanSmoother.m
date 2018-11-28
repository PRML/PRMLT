function [nu, U, llh, Ezz, Ezy] = kalmanSmoother(model, X)
% Kalman smoother (forward-backward algorithm for linear dynamic system)
% NOTE: This is the exact implementation of the Kalman smoother algorithm in PRML.
% However, this algorithm is not practical. It is numerical unstable. 
% Input:
%   X: d x n data matrix
%   model: model structure
% Output:
%   nu: q x n matrix of latent mean mu_t=E[z_t] w.r.t p(z_t|x_{1:T})
%   U: q x q x n latent covariance U_t=cov[z_t] w.r.t p(z_t|x_{1:T})
%   Ezz: q x q matrix E[z_tz_t^T]
%   Ezy: q x q matrix E[z_tz_{t-1}^T]
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu0; % prior mean
P0 = model.P0;  % prior covairance

n = size(X,2);
q = size(mu0,1);
mu = zeros(q,n);
V = zeros(q,q,n);
P = zeros(q,q,n); % C_{t+1|t}
Amu = zeros(q,n); % u_{t+1|t}
llh = zeros(1,n);

% forward
PC = P0*C';
R = C*PC+S;
K = PC/R;
mu(:,1) = mu0+K*(X(:,1)-C*mu0);
V(:,:,1) = (eye(q)-K*C)*P0;
P(:,:,1) = P0;  % useless, just make a point
Amu(:,1) = mu0; % useless, just make a point
llh(1) = logGauss(X(:,1),C*mu0,R);
for i = 2:n    
    [mu(:,i), V(:,:,i), Amu(:,i), P(:,:,i), llh(i)] = ...
        forwardUpdate(X(:,i), mu(:,i-1), V(:,:,i-1), A, G, C, S);
end
llh = sum(llh);
% backward
nu = zeros(q,n);
U = zeros(q,q,n);
Ezz = zeros(q,q,n);
Ezy = zeros(q,q,n-1);

nu(:,n) = mu(:,n);
U(:,:,n) = V(:,:,n);
Ezz(:,:,n) = U(:,:,n)+nu(:,n)*nu(:,n)';
for i = n-1:-1:1  
    [nu(:,i), U(:,:,i), Ezz(:,:,i), Ezy(:,:,i)] = ...
        backwardUpdate(nu(:,i+1), U(:,:,i+1), mu(:,i), V(:,:,i), Amu(:,i+1), P(:,:,i+1), A);
end

function [mu1, V1, Amu, P, llh] = forwardUpdate(x, mu0, V0, A, G, C, S)
k = numel(mu0);
P = A*V0*A'+G;                                               % 13.88
PC = P*C';
R = C*PC+S;
K = PC/R;                                                    % 13.92
Amu = A*mu0;
CAmu = C*Amu;
mu1 = Amu+K*(x-CAmu);                                        % 13.89
V1 = (eye(k)-K*C)*P;                                         % 13.90
llh = logGauss(x,CAmu,R);                                    % 13.91


function [nu0, U0, E00, E10] = backwardUpdate(nu1, U1, mu, V, Amu, P, A)
J = V*A'/P;                                                  % 13.102
nu0 = mu+J*(nu1-Amu);                                        % 13.100
U0 = V+J*(U1-P)*J';                                          % 13.101
E00 = U0+nu0*nu0';                                           % 13.107
E10 = U1*J'+nu1*nu0';                                        % 13.106 
