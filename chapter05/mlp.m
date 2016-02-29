function model = mlp(X, Y, h)



[D,N] = size(X);
h = [h,size(Y,1)];
L = numel(h);

maxiter = 200;
eta = 1/N;

W = cell(L);
W{1} = randn(D,h(1));

for l = 2:L
    W{l} = randn(h(l-1),h(l));
end


Z = cell(L);

for iter = 1:maxiter
%     forward
    Z{1} = sigmoid(W{1}'*X);
    for l = 2:L
        Z{l} = sigmoid(W{l}'*Z{l-1});
    end
%     backward
    E = Y-Z{L};
    df = Z{L}.*(1-Z{L});
    
    for l = L-1:-1:1
        df = Z{l}.*(1-Z{l});
        dG = df.*E;
        dW = Z{l}*dG;
        W = W+eta*dW;
        E = (W{l}*dG);
    end
end
model.W = W;