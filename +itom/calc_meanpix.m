function meanpix = calc_meanpix(image_path, indices, image_size)
    meanpix = single(zeros(image_size, image_size, 3));
    batch_size = 256;
    n = length(indices);
    for i = 1 : batch_size : n
        meta_index = i : min(i + batch_size - 1, n);
        index = indices(meta_index);
        image_batch = single(itom.load_images(image_path, index, image_size));  % [H, W, C, n]
        meanpix = meanpix + sum(image_batch, 4);
        fprintf("%d\n", i);
    end
    meanpix = meanpix / n;
end
