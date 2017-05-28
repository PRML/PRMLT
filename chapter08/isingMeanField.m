function mu = isingMeanField(J, h, epoch)
if nargin < 3
    epoch = 10;
end
[M,N] = size(h);
mu =  tanh(h);
stride = [-1,1,-M,M];
for t = 1:epoch
    for j = 1:N
        for i = 1:M
            pos = i + M*(j-1);
            ne = pos + stride;
            ne([i,i,j,j] == [1,M,1,N]) = [];
            mu(i,j) = tanh(J*sum(mu(ne)) + h(i,j));
        end
    end
end 

