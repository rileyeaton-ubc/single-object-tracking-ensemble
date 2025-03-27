function dsst_tracker(path, initRect)
    % videoPath: path to folder with image frames
    % initRect: [x, y, width, height]
    
    %% Parameters
    padding = 1.5;
    output_sigma_factor = 1/16; % For the Gaussian defined for the roi
    scale_sigma_factor = 1/4; % Defines the scale confidence
    scale_step = 1.02; % Defines how much the DSST changes scale from level to level
    num_scales = 33;
    scale_model_max_area = 512; % Limits the tracking window size
    lambda = 1e-4; % For regularization to keep weights small
    learning_rate = 0.025;
    
    %% Load frame files
    valid_exts = {'*.jpg','*.png',};
    frames = [];
    for k = 1:length(valid_exts)
        frames = [frames; dir(fullfile(path, valid_exts{k}))]; 
    end
    if isempty(frames)
        error('No image frames found in folder: %s', path);
    end
    [~, idx] = sort_nat({frames.name});
    frames = frames(idx);
    numFrames = length(frames);
    getFrame = @(i) imread(fullfile(path, frames(i).name));
    
    %fprintf('Loaded %d frames from: %s\n', numFrames, path);
    %disp({frames.name});
    
    %% Load first frame
    im = getFrame(1);
    if size(im,3) > 1
        im_disp = im; 
        im = rgb2gray(im);
    else
        im_disp = im;
    end
    %% Select initial bounding box if not provided
    if nargin < 2 || isempty(initRect)
       figure; 
       imshow(im_disp);
       title('Draw initial bounding box, then press Enter');
       h = imrect;
       initRect = round(wait(h)); % [x y w h]
       close;
    end
    %% Initialization
    pos = [initRect(2) + initRect(4)/2, initRect(1) + initRect(3)/2]; % [y, x]
    target_sz = [initRect(4), initRect(3)];
    window_sz = floor(target_sz * (1 + padding));
    output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
    yf = fft2(gaussian_shaped_labels(output_sigma, window_sz));
    cos_window = hann(window_sz(1)) * hann(window_sz(2))';
    % Scale init
    scale_factors = scale_step.^((1:num_scales) - ceil(num_scales/2));
    scale_model_sz = target_sz;
    if prod(scale_model_sz) > scale_model_max_area
       scale_model_factor = sqrt(scale_model_max_area / prod(scale_model_sz));
       scale_model_sz = floor(scale_model_sz * scale_model_factor);
    else
       scale_model_factor = 1.0;
    end
    scale_sigma = num_scales * scale_sigma_factor;
    ss = (1:num_scales) - ceil(num_scales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = fft(ys);
    scale_window = hann(num_scales)';
    base_target_sz = target_sz;
    currentScaleFactor = 1.0;

    %% Main Loop
    for frame = 1:numFrames
        im = getFrame(frame);
        if size(im,3) > 1, im_disp = im; im = rgb2gray(im); else, im_disp = im; end
        if frame == 1
            patch = get_subwindow(im, pos, window_sz);
            x = double(patch) / 255;
            x = x .* cos_window;
            xf = fft2(x);
            kf = sum(xf .* conj(xf), 3);
            alphaf = yf ./ (kf + lambda);
            zf = xf;
            scale_sample = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scale_factors, scale_model_sz);
            sf = fft(scale_sample, [], 2);
            sf_num = bsxfun(@times, ysf, conj(sf));
            sf_den = sum(sf .* conj(sf), 1);
        else
            patch = get_subwindow(im, pos, window_sz);
            x = double(patch) / 255;
            x = x .* cos_window;
            zf_new = fft2(x);
            kzf = sum(zf_new .* conj(zf), 3);
            response = real(ifft2(alphaf .* kzf));
            [row, col] = find(response == max(response(:)), 1);
            pos = pos + ([row, col] - floor(size(response)/2) - 1);
            % Scale estimation
            scale_sample = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scale_factors, scale_model_sz);
            xsf = fft(scale_sample, [], 2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
            [~, scale_ind] = max(scale_response);
            currentScaleFactor = currentScaleFactor * scale_factors(scale_ind);
            % Update
            xf = fft2(x);
            kf = sum(xf .* conj(xf), 3);
            alphaf = (1 - learning_rate) * alphaf + learning_rate * (yf ./ (kf + lambda));
            zf = (1 - learning_rate) * zf + learning_rate * xf;
            sf = fft(scale_sample, [], 2);
            sf_num = (1 - learning_rate) * sf_num + learning_rate * bsxfun(@times, ysf, conj(sf));
            sf_den = (1 - learning_rate) * sf_den + learning_rate * sum(sf .* conj(sf), 1);
        end

        % Draw bounding box
        rect_pos = [pos([2,1]) - (base_target_sz([2,1]) * currentScaleFactor)/2, ...
                   base_target_sz([2,1]) * currentScaleFactor];
        imshow(im_disp); hold on;
        rectangle('Position', rect_pos, 'EdgeColor', 'g', 'LineWidth', 2);
        title(sprintf('Frame %d / %d', frame, numFrames));
        drawnow;
    end
end

%% Helper functions
function labels = gaussian_shaped_labels(sigma, sz)
    [rs, cs] = ndgrid( (1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
    labels = exp(-0.5 * ((rs.^2 + cs.^2) / sigma^2));
end

function out = get_subwindow(im, pos, sz)
    xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
    ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
    xs(xs < 1) = 1; ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    out = im(ys, xs);
end

function out = get_scale_sample(im, pos, base_target_sz, scale_factors, scale_model_sz)
    num_scales = numel(scale_factors);
    out = zeros(prod(scale_model_sz), num_scales, 'double');
    for s = 1:num_scales
        patch_sz = floor(base_target_sz * scale_factors(s));
        patch = get_subwindow(im, pos, patch_sz);
        resized = imresize(patch, scale_model_sz, 'bilinear');
        out(:,s) = reshape(double(resized)/255, [], 1);
    end
    out = bsxfun(@times, out, hann(num_scales)');
end
function [sorted, idx] = sort_nat(cellArray)
   expr = '\d+';
   tokens = regexp(cellArray, expr, 'match');
   nums = cellfun(@(x) sscanf([x{:}], '%d'), tokens);
   [~, idx] = sort(nums);
   sorted = cellArray(idx);
end
