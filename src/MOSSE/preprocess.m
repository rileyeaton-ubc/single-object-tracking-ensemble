function img_out = preprocess(img)
    % preprocess applies logarithmic compression, normalization, and windowing.
    [r, c] = size(img);
    win = window2(r, c, @hann);
    eps_val = 1e-5;
    img = log(double(img) + 1);
    img = (img - mean(img(:))) / (std(img(:)) + eps_val);
    img_out = img .* win;
end