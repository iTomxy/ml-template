% normalize X per row
% Usage:
%   p_normalize(X [, p])
% Input:
%   X: [n, d], row vector set
% Output:
%   X_norm: [n, d], with X_norm(i,:) = normalize(X, p)
function X_norm = p_normalize(X, varargin)
    args = inputParser;
    addOptional(args, 'p', 2, @(p) isa(p, 'numeric'));
    parse(args, varargin{:});
    p = args.Results.p;

    r_norm = sum(abs(X) .^ p, 2) .^ (1 / p);
    r_norm = repmat(r_norm, 1, size(X, 2));
    X_norm = X ./ r_norm;
end
