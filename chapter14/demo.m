clear all;
%% Generate Data

d = 1;
k = 3;
n = 500;
W = randn(d+1,k);

[x, label] = kmeansrnd(d, k, n);
X = [x; ones(1,n)];
y = zeros(1,n);
for j = 1:k
    idx = (label == j);
    y(idx) = W(:,j)'*X(:,idx);
end

plot(x,y,'.');

[model,llh] = mixLinReg(X, y, 2);
plot(llh);
% 
% figure();
% subplot(1,3,1);
% plot(x1,y1,'r*'); hold on;
% plot(x2,y2,'bx');
% hold off;
% subplot(1,3,2);
% plot(X,y,'r*');
% % subplot(1,3,3);
% hold on;
% plot(x1,x1*model.W(1,1) + model.W(2,1),'r-');hold on;
% plot(x2,x2*model.W(1,2) + model.W(2,2),'b-');
% hold off;
