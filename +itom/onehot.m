% convert sparse class IDs to one-hot label vectors
% Input:
%   vec: [1, n] row or [n, 1] column vector
%   n_class: int, # of classes
% Output:
%   L: [n, n_class] one-hot class label vectors
function L = onehot(vec, n_class)
    assert(isvector(vec), "`vec` must be a vector");
    assert(isa(vec, 'integer'), "`vec` must be integer");
    if isrow(vec)
        vec = vec';  % -> column vector
    end
    I = eye(n_class);
    L = I(vec, :);
end
