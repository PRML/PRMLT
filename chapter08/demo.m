clear; close all;
% load letterA.mat;
% X = A;
load letterX.mat
%% Original image
epoch = 50;
J = 1;   % ising parameter
sigma = 1; % noise level

img = double(X);
img = sign(img-mean(img(:)));

figure;
subplot(2,3,1);
imagesc(img);
title('Original image');
axis image;
colormap gray;
%% Noisy image
y = img + sigma*randn(size(img)); % noisy signal
subplot(2,3,2);
imagesc(y);
title('Noisy image');
axis image;
colormap gray;
%% Mean Field
[A, nodePot, edgePot] = im2mrf(y, sigma, J);
[nodeBel, edgeBel, lnZ] = meanField(A, nodePot, edgePot, epoch);
lnZ0 = gibbsEnergy(nodePot, edgePot, nodeBel, edgeBel);
lnZ1 = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel);
maxdiff(lnZ0, lnZ(end))
maxdiff(lnZ0, lnZ1)

subplot(2,3,3);
imagesc(reshape(nodeBel(1,:),size(img)));
title('MF');
axis image;
colormap gray;
%% Belief Propagation
% [nodeBel,edgeBel] = belProp(A, nodePot, edgePot, epoch);
% 
% [nodeBel0,edgeBel0] = belProp0(A, nodePot, edgePot, epoch);
% maxdiff(nodeBel,nodeBel0)
% maxdiff(edgeBel,edgeBel0)
% 
% subplot(2,3,4);
% imagesc(reshape(nodeBel(1,:),size(img)));
% title('BP');
% axis image;
% colormap gray;
% %% Expectation Propagation
% [nodeBel,edgeBel] = expProp(A, nodePot, edgePot, epoch);
% 
% lnZ0 = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel);
% 
% [nodeBel0,edgeBel0] = expProp0(A, nodePot, edgePot, epoch);
% maxdiff(nodeBel,nodeBel0)
% maxdiff(edgeBel,edgeBel0)
% 
% subplot(2,3,5);
% imagesc(reshape(nodeBel(1,:),size(img)));
% title('EP');
% axis image;
% colormap gray;
% %% EP-BP
% [nodeBel,edgeBel] = expBelProp(A, nodePot, edgePot, epoch);
% 
% [nodeBel0,edgeBel0] = expBelProp0(A, nodePot, edgePot, epoch);
% maxdiff(nodeBel,nodeBel0)
% maxdiff(edgeBel,edgeBel0)
% 
% subplot(2,3,6);
% imagesc(reshape(nodeBel(1,:),size(img)));
% title('EBP');
% axis image;
% colormap gray;
