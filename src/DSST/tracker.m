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
end

function [sorted, idx] = sort_nat(cellArray)
   expr = '\d+';
   tokens = regexp(cellArray, expr, 'match');
   nums = cellfun(@(x) sscanf([x{:}], '%d'), tokens);
   [~, idx] = sort(nums);
   sorted = cellArray(idx);
end
