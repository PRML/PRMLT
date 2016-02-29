% The XOR Example - Batch-Mode Training
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br
close all;
p = 2;
H = 4;
m = 1;

mu = .75;



X = [0 0 1 1;0 1 0 1];
D = [0 1 1 0];

[Wx,Wy,MSE]=trainMLP2(X,D,H);
model = mlp(X,D,H);

semilogy(MSE);
% 
% disp(['D = [' num2str(D) ']']);
% 
% Y = runMLP(X,Wx,Wy);
% 
% disp(['Y = [' num2str(Y) ']']);

