function dsst_tracker(path, initRect)
    % videoPath: path to folder with image frames
    % initRect: [x, y, width, height]
    
    %% Parameters
    padding = 1.5;
    output_sigma_factor = 1/16; % For the Gaussian defined for the roi
    scale_sigma_factor = 1/4; % Defines the scale confidence
    scale_step = 1.02; % Defines how much the DSST changes scale from level to level
    num_scales = 33;
    scale_model_max_area = 256; % Limits the tracking window size
    lambda = 1e-4; % For regularization to keep weights small
    learning_rate = 0.025;
    
    %% Load Ground Truth file
    gtFile = fullfile(path, "groundtruth_rect.txt");
    if ~isfile(gtFile)
        error('Ground truth file not found: %s', gtFile);
    end
    groundTruth = readmatrix(gtFile);
    
    %% Load frame files
    valid_exts = {'*.jpg','*.png',};
    frames = [];
    for k = 1:length(valid_exts)
        frames = [frames; dir(fullfile(path + "\img", valid_exts{k}))]; 
    end
    if isempty(frames)
        error('No image frames found in folder: %s', path);
    end
    [~, idx] = sort_nat({frames.name});
    frames = frames(idx);
    numFrames = length(frames);
    getFrame = @(i) imread(fullfile(path + "\img", frames(i).name));

    iouscores = zeros(numFrames, 1);
    centerErrors = zeros(numFrames,1);
    predictedBoxes = zeros(numFrames, 4);
    
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
    pos = [initRect(1) + initRect(3)/2, initRect(2) + initRect(4)/2]; % [x_center, y_center]
    target_sz = [initRect(3), initRect(4)]; % [w,h]
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
        if (size(im,3) > 1)
            im_disp = im; 
            im = rgb2gray(im); 
        else 
            im_disp = im; 
        end

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
        topLeft = pos - (target_sz * currentScaleFactor) / 2;
        rect_pos = [topLeft(1), topLeft(2), target_sz(1) * currentScaleFactor, target_sz(2) * currentScaleFactor];
        predictedBoxes(frame, :) = rect_pos;

        gtRect = groundTruth(frame, :);
        gtRect(1:2) = gtRect(1:2)+1;
        iouscores(frame) = computeIoU(rect_pos, gtRect); 

        predCenter = [rect_pos(1) + rect_pos(3)/2, rect_pos(2) + rect_pos(4)/2];
        gtCenter = [groundTruth(frame,1) + groundTruth(frame,3)/2, ...
                groundTruth(frame,2) + groundTruth(frame,4)/2];
        centerErrors(frame) = norm(predCenter - gtCenter);

        imshow(im_disp); hold on;
        %draw_rect = [rect_pos(2), rect_pos(1), rect_pos(4), rect_pos(3)];
        rectangle('Position', rect_pos, 'EdgeColor', 'r', 'LineWidth', 2);
        rectangle('Position', gtRect, 'EdgeColor', 'g', 'LineWidth', 2);
        legend('Tracker (Red)', 'Ground Truth (Green)');
        title(sprintf('Frame %d / %d | IOU: %.2f | Size : %d %d', frame, numFrames, iouscores(frame), size(im_disp,1), size(im_disp,2)));
        drawnow;
    end
    meanIoU = mean(iouscores);
    fprintf('Tracking Accuracy (Mean IoU): %.4f\n', meanIoU);
    
    figure;
    thresholds = 0:1:50;  % Center error thresholds (0 to 50 pixels)
    precision = arrayfun(@(t) mean(centerErrors < t), thresholds);
    plot(thresholds, precision, 'LineWidth', 2);
    xlabel('Center Error Threshold (pixels)'); ylabel('Precision');
    title('Precision Plot');
    grid on;
    
    figure;
    iouThresholds = 0:0.05:1;  % IoU thresholds (0 to 1)
    successRate = arrayfun(@(t) mean(iouscores > t), iouThresholds);
    plot(iouThresholds, successRate, 'LineWidth', 2);
    xlabel('IoU Threshold'); ylabel('Success Rate');
    title('Success Plot');
    grid on;
    
    outputFile = fullfile(".", 'dsst_tracking_results.txt');
    writematrix(predictedBoxes, outputFile, 'Delimiter', 'tab');
    
    fprintf('Tracking results saved to: %s\n', outputFile);

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

function iou = computeIoU(boxA, boxB)
    % boxA and boxB: [x, y, width, height]
    xA = max(boxA(1), boxB(1));  
    yA = max(boxA(2), boxB(2)); 
    xB = min(boxA(1) + boxA(3), boxB(1) + boxB(3)); 
    yB = min(boxA(2) + boxA(4), boxB(2) + boxB(4)); 
    interArea = max(0, xB - xA) * max(0, yB - yA);
    boxAArea = boxA(3) * boxA(4);  
    boxBArea = boxB(3) * boxB(4);
    
    iou = interArea / (boxAArea + boxBArea - interArea);
end
