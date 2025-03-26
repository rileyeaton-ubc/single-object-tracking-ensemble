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
end

function [sorted, idx] = sort_nat(cellArray)
   expr = '\d+';
   tokens = regexp(cellArray, expr, 'match');
   nums = cellfun(@(x) sscanf([x{:}], '%d'), tokens);
   [~, idx] = sort(nums);
   sorted = cellArray(idx);
end
