% Jaccard similarity of labels
% Usage:
%   jaccard(L [, L2])
% Input:
%   L, L2: [n, c], [m, c], in {0, 1}
% Output:
%   S: [n, m], with S(i,j) = jaccard(l_i, l2_j)
function S = jaccard(L, varargin)
    args = inputParser;
    addOptional(args, 'L2', L, @(L2) (size(L, 2) == size(L2, 2)));
    parse(args, varargin{:});
    L2 = args.Results.L2;

    numer = L * L2';  % [n, m]
    L_cube = reshape(L, [size(L, 1), 1, size(L, 2)]);  % [n, 1, c]
    L_cube = repmat(L_cube, 1, size(L2, 1), 1);  % [n, m, c]
    L2_cube = reshape(L2, [1, size(L2, 1), size(L2, 2)]);  % [1, m, c]
    L2_cube = repmat(L2_cube, size(L, 1), 1, 1);  % [n, m, c]
    denom = sum(int32(L_cube + L2_cube > 0), 3);  % [n, m]
    denom(0 == denom) = 1;
    S = single(numer) ./ single(denom);
end
