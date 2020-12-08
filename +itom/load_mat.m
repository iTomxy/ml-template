function data = load_mat(path, key)
    st = load(path, key);
    data = st.(key);
end
