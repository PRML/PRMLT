close all;
%% generate data
clear; 
d = 2;
k = 4;
n = 50;

A = [1 0 1 0; 
     0 1 0 1;
     0 0 1 0;
     0 0 0 1]; 
G = 0.001*eye(k);
 
C = [1 0 0 0;
     0 1 0 0];
S = eye(d);

mu0 = [8; 10; 1; 0];
P0 = eye(k);

model.A = A;
model.G = G;
model.C = C;
model.S = S;
model.mu0 = mu0;
model.P0 = P0;

[z,x] = ldsRnd(model, n);
figure;
hold on
plot(x(1,:), x(2,:), 'ro');
plot(z(1,:), z(2,:), 'b*-');
legend('observed', 'latent')
axis equal
hold off

%% filter
[mu, V, llh] = kalmanFilter(model, x);
figure
hold on
plot(x(1,:), x(2,:), 'ro');
plot(mu(1,:), mu(2,:), 'b*-');
legend('observed', 'filtered')
axis equal
hold off

%% smoother
[nu, U, llh] = kalmanSmoother(model, x);
figure
hold on
plot(x(1,:), x(2,:), 'ro');
plot(nu(1,:), nu(2,:), 'b*-');
legend('observed', 'smoothed')
axis equal
hold off

%% EM
[model, llh] = ldsEm(x,model);
nu = kalmanSmoother(model, x);
figure
hold on
plot(x(1,:), x(2,:), 'ro');
plot(nu(1,:), nu(2,:), 'b*-');
legend('observed', 'smoothed with fitted model')
axis equal
hold off
figure;
plot(llh);
