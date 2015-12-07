
clear; close all;
k = 2;
n = 1000;
[X,t] = rndKCluster(2,k,n);

[x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
[w, llh] = logitReg(X,t-1,0.0001);
plot(llh);
figure;
spread(X,t);

y = w(1)*x1+w(2)*x2;

hold on;
contour(x1,x2,y,1);
hold off;
%%
clear; close all;
k = 4;
n = 1000;
[X,t] = rndKCluster(2,k,n);

[x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));
[model, llh] = mnReg(X,t, 1e-4, 1);
plot(llh);
figure;
spread(X,t);

W = model.W;
y = w(1)*x1+w(2)*x2+w(3);

hold on;
contour(x1,x2,t,1);
hold off;