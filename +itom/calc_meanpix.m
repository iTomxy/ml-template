% image_path: str/char, assuming that file names are in format `<index>.jpg`
% indices: 1-base int vector
% image_size: int or [int int]
function meanpix = calc_meanpix(image_path, indices, image_size)
    meanpix = single(zeros(image_size, image_size, 3));
    batch_size = 256;
    n = length(indices);
    for i = 1 : batch_size : n
        meta_index = i : min(i + batch_size - 1, n);
        index = indices(meta_index);
        img_files = {};
        for k = 1 : length(index)
            idx = index(k);
            img_files(k) = {char(fullfile(image_path, sprintf("%d.jpg", idx)))};
        end
        image_batch = single(itom.load_images(img_files, image_size));  % [H, W, C, n]
        meanpix = meanpix + sum(image_batch, 4);
        fprintf("%d\n", i);
    end
    meanpix = meanpix / n;
end
