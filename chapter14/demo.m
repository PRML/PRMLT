clear all;
%% Generate Data
W = randn(2,2);
w1 = W(1,1);
b1 = W(1,2);
w2 = W(2,1);
b2 = W(2,2);

x = linspace(-5,5,50);
x1 = x + randn(size(x)) * 0.001;
x1(x1 < -3 | x1 >  3) = [];
x2 = x + randn(size(x)) * 0.001;
x2(x2 < 3  & x2 > -3) = []; 
y1 = w1 * x1 + b1 + 5;
y2 = w2 * x2 + b2 - 5;

X = [x1 x2];
y = [y1 y2];

[model,llh,R] = mixLinRegress(X, y, 2);

figure();
subplot(1,3,1);
plot(x1,y1,'r*'); hold on;
plot(x2,y2,'bx');
hold off;
subplot(1,3,2);
plot(X,y,'r*');
% subplot(1,3,3);
hold on;
plot(x1,x1*model.W(1,1) + model.W(2,1),'r-');hold on;
plot(x2,x2*model.W(1,2) + model.W(2,2),'b-');
hold off;
subplot(1,3,3);
plot(llh);