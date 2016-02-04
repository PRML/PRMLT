function t = adaboostBinPred(model,X)
Alpha = model.alpha;
Theta = model.theta;
M = size(Alpha,2);
t = zeros(1,size(X,2));
for m = 1:M
    c = Theta(:,:,m);
    [~,y] = min(sqdist(c,X),[],1);
    y(y==1) = -1;
    y(y==2) = 1;
    t = t+Alpha(m)*y;
end
t = sign(t);
t(t==-1) = 0;