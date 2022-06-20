% function image_batch = load_images(image_path, indices, image_size)
%     image_files = {};
%     for i = 1 : length(indices)
%         idx = indices(i);
%         image_files(i) = {char(fullfile(image_path, sprintf("%d.jpg", idx)))};
%     end
%     img = vl_imreadjpeg(image_files, 'NumThreads', 4, ...
%         'Resize', [image_size, image_size], 'Pack', 'Interpolation', 'bilinear');
%     image_batch = img{1, 1};  % (H, W, C, n)
% end

%% (2022.6.18) new image loading API
% image_files: cell, paths of images to load
% image_size: int or [int int]
function image_batch = load_images(image_files, image_size)
    if isscalar(image_size)
        image_size = [image_size, image_size];
    else
        assert(isvector(image_size), "* Should be scalar or 2D vector");
        assert(length(image_size) == 2, ...
            sprintf("* dimension error: [%s]", num2str(size(image_size))));
    end % if scalar
    img = vl_imreadjpeg(image_files, 'NumThreads', 4, ...
        'Resize', image_size, 'Pack', 'Interpolation', 'bilinear');
    image_batch = img{1, 1};  % (H, W, C, n)
end
