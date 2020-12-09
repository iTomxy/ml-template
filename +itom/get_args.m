function args = get_args(data_set)
    args = struct();
    P = "/home/tom/codes/reimpl.DCMH";
    args.dataset = data_set;
    args.log_path = fullfile(P, "log.matlab");
    args.my_data = 1;
    args.my_meanpix = 1;
    args.whole_image = 0;
    args.sample_ret = 5000;
    args.test_per = 50;
    args.image_size = 224;

    if args.my_data
        DATA_P = fullfile(P, "data", data_set);
    else
        DATA_P = fullfile(P, "data", strcat(data_set, ".jqy"));
    end

    if strcmp(data_set, "nuswide-tc21") == 1
        args.split = "nuswide-tc21.100pc.500pc";
        args.split_path = fullfile(DATA_P, args.split);
        if args.my_data
            args.label_file = fullfile(DATA_P, "labels.tc-21.mat");
            args.text_file = fullfile(DATA_P, "texts.AllTags1k.mat");
        else  % original data
            args.label_file = fullfile(DATA_P, "nus-wide-tc21-lall.mat");
            args.text_file = fullfile(DATA_P, "nus-wide-tc21-yall.mat");
            args.image_file = fullfile(DATA_P, "nus-wide-tc21-iall.mat"); % (195834, 3, 224, 224)
        end
        args.image_path = fullfile(DATA_P, "images");
        args.meanpix_file = fullfile(args.split_path, sprintf("avgpix.%s.mat", args.split));  % (224, 224, 3)
    end % if data_set
end
