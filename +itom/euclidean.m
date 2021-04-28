function D = euclidean(X, varargin)
% (square) euclidean distance
% euclidean(X [, Y] [, 'square', 0/1])
% X, Y: [n, d], [m, d], feature vectors
% square: 0/1, square the euclidean distance or not, default = 1
% D: [n, m], with D(i,j) = euc_dist(xi, yj)
% -----------------------------------------
    args = inputParser;
    addOptional(args, 'Y', 1, @(Y) (size(X, 2) == size(Y, 2)));
    addParameter(args, 'square', 1);
    parse(args, varargin{:});
    Y = args.Results.Y;

    if isscalar(Y)  % only `X`
        xTy = X * X';  % [n, n]
        xTx = repmat(diag(xTy), 1, size(xTy, 2));
        yTy = xTx';
    else  % `X` and `Y`
        xTy = X * Y';  % [n, m]
        xTx = repmat(sum(X .* X, 2), 1, size(xTy, 2));  % [n, m]
        yTy = repmat(sum(Y .* Y, 2)', size(xTy, 1), 1);  % [n, m]
    end

    D = xTx - 2 * xTy + yTy;
    if 0 == args.Results.square
        D = sqrt(D);
    end
end
