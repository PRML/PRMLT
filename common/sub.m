function B = sub(A, varargin)
% submat(A,i,j,k) = A(i;j;k)
assert(ndims(A)==numel(varargin));
sz = cellfun(@numel,varargin);
IDX = cell(1,ndims(A));
for i = 1:ndims(A)
    idx = varargin{i};
    shape = ones(1,ndims(A));
    shape(i) = sz(i);
    idx = reshape(idx,shape);
    shape = sz;
    shape(i) = 1;
    idx = repmat(idx,shape);
    IDX{i} = idx(:);
end
B = reshape(A(sub2ind(size(A),IDX{:})),sz);