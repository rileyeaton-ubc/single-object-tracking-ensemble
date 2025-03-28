function warped = rand_warp(img)
    % rand_warp applies a random affine transformation to the image.
    % It pads the image to reduce boundary issues, applies random rotation
    % and scaling, and then crops back to the original size.
    
    pad = 10;  % Padding amount
    img_padded = padarray(img, [pad, pad], 'replicate', 'both');
    
    % Define random transformation parameters
    a = -180/16;
    b = 180/16;
    r = a + (b - a) * rand;  % Random rotation angle in degrees
    scale = 1 - 0.1 + 0.2 * rand;  % Random scale factor between 0.9 and 1.1
    
    % Apply rotation and scaling on padded image
    warped_pad = imrotate(img_padded, r, 'bilinear', 'crop');
    warped_pad = imresize(warped_pad, scale);
    
    % Crop back to original image size
    sz = size(img);
    [h_pad, w_pad] = size(warped_pad);
    center_row = round(h_pad / 2);
    center_col = round(w_pad / 2);
    
    start_row = max(1, center_row - floor(sz(1) / 2));
    start_col = max(1, center_col - floor(sz(2) / 2));
    if start_row + sz(1) - 1 > h_pad
        start_row = h_pad - sz(1) + 1;
    end
    if start_col + sz(2) - 1 > w_pad
        start_col = w_pad - sz(2) + 1;
    end
    crop_rect = [start_col, start_row, sz(2) - 1, sz(1) - 1];
    warped = imcrop(warped_pad, crop_rect);
    
    % If cropping fails, use the central crop from the padded image.
    if isempty(warped)
        warped = imcrop(warped_pad, [floor(w_pad/2)-floor(sz(2)/2), floor(h_pad/2)-floor(sz(1)/2), sz(2)-1, sz(1)-1]);
    end
end