function param = split(args)
    % shift: 0-base -> 1-base
    if strcmp(args.dataset, "nuswide-tc21") == 1
        param.indexTrain = itom.load_mat(fullfile(args.split_path, "idx_labeled.mat"), "index")' + 1;
        param.indexQuery = itom.load_mat(fullfile(args.split_path, "idx_test.mat"), "index")' + 1;
        param.indexRetrieval = itom.load_mat(fullfile(args.split_path, "idx_ret.mat"), "index")' + 1;
    end
end
