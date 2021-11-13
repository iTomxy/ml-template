function bool = inf_nan(varargin)
    bool = 0;
    for i = 1 : nargin
        x = reshape(varargin{i}, [], 1);
        bool = bool | any(isinf(x)) | any(isnan(x));
        if bool
            break;
        end
    end
end

