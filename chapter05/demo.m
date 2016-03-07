clear; close all;
h = 4;
X = [0 0 1 1;0 1 0 1];
Y = [0 1 1 0];
[model,mse] = mlp(X,Y,h);
plot(mse);
