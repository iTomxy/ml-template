function image_batch = load_images(image_path, indices, image_size)
    image_files = {};
    for i = 1 : length(indices)
        idx = indices(i);
        image_files(i) = {char(fullfile(image_path, sprintf("%d.jpg", idx)))};
    end
    img = vl_imreadjpeg(image_files, 'NumThreads', 4, ...
        'Resize', [image_size, image_size], 'Pack', 'Interpolation', 'bilinear');
    image_batch = img{1, 1};  % (H, W, C, n)
end
