%% regression
% d = 100;
% beta = 1e-1;
% X = rand(1,d);
% w = randn;
% b = randn;
% t = w'*X+b+beta*randn(1,d);
% 
% x = linspace(min(X)-1,max(X)+1,d);   % test data
% 
% [model,llh] = rvmRegFp(X,t);
% figure
% plot(llh);
% [y, sigma] = linPred(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off
%%
% [model,llh] = rvmRegEm(X,t);
% figure
% plot(llh);
% [y, sigma] = linPred(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off
%%
% [model,llh] = rvmRegSeq(X,t);
% figure
% plot(llh);
% [y, sigma] = linPred(x,model,t);
% figure;
% hold on;
% plotBand(x,y,2*sigma);
% plot(X,t,'o');
% plot(x,y,'r-');
% hold off