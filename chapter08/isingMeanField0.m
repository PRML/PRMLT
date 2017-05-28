function mu = isingMeanField0(J, h, epoch)
% use padding trick
if nargin < 3
    epoch = 10;
end
mu = zeros(size(h)+2);                        % padding
[m,n] = size(mu);
mu(2:m-1,2:n-1) = tanh(h);               % init
stride = [-1,1,-m,m];
for t = 1:epoch
    for j = 2:n-1
        for i = 2:m-1
            ne = i + m*(j-1) + stride;
            mu(i,j) = tanh(J*sum(mu(ne))+h(i-1,j-1));
        end
    end
end
mu = mu(2:m-1,2:n-1);