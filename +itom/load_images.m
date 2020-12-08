function image_batch = load_images(IMAGE_P, indices)
    % shift: 1-base -> 0-base
    indices = indices - 1;
    image_batch = [];
    for i = 1 : length(indices)
        idx = indices(i);
        img_p = fullfile(IMAGE_P, strcat(num2str(idx), ".mat"));
        % img = cell2mat(struct2cell(load(img_p)));
        % img_ = load(img_p, "image");
        % img = img_.image;
        img = itom.load_mat(img_p, "image");
        image_batch(i, :, :, :) = img;
    end
end
