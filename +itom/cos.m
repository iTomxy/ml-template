function S = cos(X, varargin)
% cosine similarity
% calling: cos(X [, Y])
% Input:
%   X, Y: [n, d], [m, d], feature vectors
% Output:
%   S: [n, m], with D(i,j) = dot(X(i,;), Y(j,:)) / (||X(i,:)|| * ||Y(j,:)||)
% --------------------------------------------------------------------------
    args = inputParser;
    addOptional(args, 'Y', 1, @(Y) (size(X, 2) == size(Y, 2)));
    parse(args, varargin{:});
    Y = args.Results.Y;

    X_norm = itom.p_normalize(X, 2);
    if isscalar(Y)  % only `X`
        Y_norm = X_norm;
    else  % `X` and `Y`
        Y_norm = itom.p_normalize(Y, 2);
    end
    S = X_norm * Y_norm';
end
