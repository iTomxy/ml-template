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

function image_batch = load_images(indices, id2img, image_size)
% (2021.11.13) new image loading API
% indices: [n], 1-base index vector
% id2img: cell, maps index to corresponding absolute image file path
% image_size: int
%-------------------------------------------------------------------
    image_files = {};
    for i = 1 : length(indices)
        idx = indices(i);
        image_files(i) = {char(id2img{idx})};
    end
    img = vl_imreadjpeg(image_files, 'NumThreads', 4, ...
        'Resize', [image_size, image_size], 'Pack', 'Interpolation', 'bilinear');
    image_batch = img{1, 1};  % (H, W, C, n)
end
