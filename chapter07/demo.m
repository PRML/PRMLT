clear; close all;

%% regression
n = 100;
beta = 1e-1;
X = rand(1,n);
w = randn;
b = randn;
t = w'*X+b+beta*randn(1,n);

x = linspace(min(X)-1,max(X)+1,n);   % test data
%%
[model,llh] = regressRvmEbFp(X,t);
figure
plot(llh);
[y, sigma] = linInfer(x,model,t);
figure;
hold on;
plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off
%%
[model,llh] = regressRvmEbEm(X,t);
figure
plot(llh);
[y, sigma] = linInfer(x,model,t);
figure;
hold on;
plotBand(x,y,2*sigma);
plot(X,t,'o');
plot(x,y,'r-');
hold off

%% classification
k = 2;
d = 2;
n = 1000;
[X,t] = rndKCluster(d,k,n);
[x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));

%%
[model, llh] = classRvmEbFp(X,t-1);
figure
plot(llh);
figure;
spread(X,t);

w = model.w;
y = w(1)*x1+w(2)*x2+w(3);
hold on;
contour(x1,x2,y,1);
hold off;
%%
[model, llh] = classRvmEbEm(X,t-1);
figure
plot(llh);
figure;
spread(X,t);

w = model.w;
y = w(1)*x1+w(2)*x2+w(3);
hold on;
contour(x1,x2,y,1);
hold off;