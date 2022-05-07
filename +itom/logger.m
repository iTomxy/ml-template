% https://ch.mathworks.com/matlabcentral/answers/516548-using-fprintf-to-write-to-multiple-files-simultaneously#answer_424960
function logger(fid, varargin)
    fprintf(varargin{:});
    fprintf(fid, varargin{:});
end
