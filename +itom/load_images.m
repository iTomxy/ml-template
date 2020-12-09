function image_batch = load_images(path, indices, image_size)
    image_path = {};
    for i = 1 : length(indices)
        image_path(i) = {char(fullfile(path, sprintf("%d.jpg", i)))};
    end
    img = vl_imreadjpeg(image_path, 'NumThreads', 4, 'Resize', [image_size, image_size], 'Pack');
    image_batch = img{1, 1};  % (H, W, C, n)
end
