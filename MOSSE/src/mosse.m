resultsFile = fullfile(pwd, 'AUC_results.txt');
fid = fopen(resultsFile, 'w');
if fid == -1
    error('Could not open results file for writing.');
end
fprintf(fid, 'AUC Results for Each Dataset:\n\n');

% Prepare a cell array to store success plot data for all datasets
successData = {};

% Base directory containing multiple datasets (e.g., Bird1, Bird2, BlurCar1, etc.)
datadir = fullfile('..', 'data', 'OTB2015', 'OTB2015');

% Get all dataset folders (ignoring . and ..)
allFolders = dir(datadir);
allFolders = allFolders([allFolders.isdir]);  % keep only directories
allFolders = allFolders(~ismember({allFolders.name}, {'.', '..'}));

% Loop over each dataset folder
for d = 1:length(allFolders)
    dataset = allFolders(d).name;
    fprintf('Processing dataset: %s\n', dataset);
    pathDataset = fullfile(datadir, dataset);
    img_path = fullfile(pathDataset, 'img');
    
    % Get image files from current dataset folder (any .jpg pattern)
    D = dir(fullfile(img_path, '*.jpg'));
    if isempty(D)
        error('No image files found in the directory for dataset %s.', dataset);
    end
    % Sort filenames alphabetically and build full file paths
    sortedNames = sort({D.name});
    seq_len = length(sortedNames);
    img_files = cell(seq_len, 1);
    for k = 1:seq_len
        img_files{k} = fullfile(img_path, sortedNames{k});
    end
    
    % Read ground truth bounding boxes from file
    % (Assumes each line in groundtruth_rect.txt is: x,y,width,height)
    gt_file = fullfile(pathDataset, 'groundtruth_rect.txt');
    if exist(gt_file, 'file')
        labels = dlmread(gt_file, ',');
    else
        error('Ground truth file not found for dataset %s.', dataset);
    end
    
    % Use the ground truth for the first frame as the initial ROI
    rect = labels(1, :);  % [x, y, width, height]
    center = [rect(2) + rect(4)/2, rect(1) + rect(3)/2];
    
    % Plot Gaussian on the first frame
    im = imread(img_files{1});
    sigma = 100;
    gsize = size(im);
    [R, C] = ndgrid(1:gsize(1), 1:gsize(2));
    g = gaussC(R, C, sigma, center);
    g = mat2gray(g);
    
    % Prepare initial training sample
    if size(im, 3) == 3 
        img = rgb2gray(im); 
    else
        img = im;
    end
    % Ensure the rectangle is within image bounds
    [imgH, imgW, ~] = size(img);
    rect(1) = max(1, rect(1));
    rect(2) = max(1, rect(2));
    rect(3) = min(rect(3), imgW - rect(1));
    rect(4) = min(rect(4), imgH - rect(2));
    
    img_crop = imcrop(img, rect);
    g_crop = imcrop(g, rect);
    G = fft2(g_crop);
    height = size(g_crop, 1);
    width = size(g_crop, 2);
    
    if isempty(img_crop)
        error('Cropped image is empty in the initial frame for dataset %s. Check ROI.', dataset);
    end
    
    fi = preprocess(imresize(img_crop, [height width]));
    Ai = (G .* conj(fft2(fi)));
    Bi = (fft2(fi) .* conj(fft2(fi)));
    N = 128;
    for i = 1:N
        fi = preprocess(imresize(rand_warp(img_crop), [height width]));
        Ai = Ai + (G .* conj(fft2(fi)));
        Bi = Bi + (fft2(fi) .* conj(fft2(fi)));
    end
    
    % MOSSE online training regimen
    eta = 0.125;
    fig = figure('Name', ['MOSSE: ' dataset]);
    general_results_dir = fullfile(pwd, 'result');
    if ~exist(general_results_dir, 'dir')
        mkdir(general_results_dir);
    end
    results_dir = fullfile(general_results_dir, dataset);
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    
    % Initialize IoU array (for frames with available ground truth)
    ious = nan(seq_len, 1);
    
    for i = 1:seq_len
        im_disp = imread(img_files{i});
        img_frame = im_disp;
        if size(img_frame, 3) == 3
            img_frame = rgb2gray(img_frame);
        end
        
        if i == 1
            % Use initial training already computed
            Ai = eta .* Ai;
            Bi = eta .* Bi;
        else
            Hi = Ai ./ Bi;
            
            % --- First crop for computing correlation response ---
            [frameH, frameW, ~] = size(img_frame);
            rect(1) = max(1, rect(1));
            rect(2) = max(1, rect(2));
            if rect(1) + rect(3) > frameW
                rect(3) = frameW - rect(1);
            end
            if rect(2) + rect(4) > frameH
                rect(4) = frameH - rect(2);
            end
            fi = imcrop(img_frame, rect);
            if isempty(fi)
                warning('Frame %d: imcrop returned empty. Skipping update for this frame.', i);
                continue; % Skip update if crop is empty
            end
            fi = preprocess(imresize(fi, [height width]));
            gi = uint8(255 * mat2gray(ifft2(Hi .* fft2(fi))));
            maxval = max(gi(:));
            [P, Q] = find(gi == maxval);
            dx = mean(P) - height/2;
            dy = mean(Q) - width/2;
            
            % --- Update predicted bounding box ---
            rect = [rect(1) + dy, rect(2) + dx, width, height];
            [frameH, frameW, ~] = size(img_frame);
            rect(1) = max(1, rect(1));
            rect(2) = max(1, rect(2));
            if rect(1) + rect(3) > frameW
                rect(3) = frameW - rect(1);
            end
            if rect(2) + rect(4) > frameH
                rect(4) = frameH - rect(2);
            end
            fi = imcrop(img_frame, rect);
            if isempty(fi)
                warning('Frame %d: Updated imcrop returned empty. Skipping update for this frame.', i);
                continue;
            end
            fi = preprocess(imresize(fi, [height width]));
            Ai = eta .* (G .* conj(fft2(fi))) + (1 - eta) .* Ai;
            Bi = eta .* (fft2(fi) .* conj(fft2(fi))) + (1 - eta) .* Bi;
        end
        
        % Visualization: Draw ground truth (red) and predicted (yellow) bounding boxes
        if i <= size(labels,1)
            gt_rect = labels(i, :);
            ious(i) = computeIoU(rect, gt_rect);
            im_disp = insertShape(im_disp, 'Rectangle', gt_rect, 'LineWidth', 3, 'Color', 'red');
        end
        im_disp = insertShape(im_disp, 'Rectangle', rect, 'LineWidth', 3, 'Color', 'yellow');
        text_str = sprintf('Frame: %d', i);
        im_disp = insertText(im_disp, [1 1], text_str, 'FontSize', 15, 'BoxColor', 'green', 'BoxOpacity', 0.4, 'TextColor', 'white');
        
        % -- Output code for saving and displaying frames (commented out) --
        % out_file = fullfile(results_dir, sprintf('%04d.jpg', i));
        % imwrite(im_disp, out_file);
        % imshow(im_disp);
        % drawnow;
    end
    
    % Compute Success Plot and AUC for the current dataset
    thresholds = 0:0.05:1;
    successRates = zeros(size(thresholds));
    validFrames = ~isnan(ious);
    numValid = sum(validFrames);
    for j = 1:length(thresholds)
        successRates(j) = sum(ious(validFrames) >= thresholds(j)) / numValid;
    end
    auc = trapz(thresholds, successRates) / (max(thresholds) - min(thresholds));
    fprintf('Dataset %s: AUC (Area Under the Success Plot Curve) = %.4f\n', dataset, auc);
    fprintf(fid, 'Dataset %s: AUC = %.4f\n', dataset, auc);
    
    % Optionally, plot the success plot for the current dataset 
    % figure;
    % plot(thresholds, successRates, '-o', 'LineWidth', 2);
    % xlabel('Overlap Threshold');
    % ylabel('Success Rate');
    % title(sprintf('Success Plot for %s', dataset));
    % grid on;
    
    % Store data for combined AUC graph
    successData{end+1} = struct('dataset', dataset, 'thresholds', thresholds, 'successRates', successRates, 'auc', auc);
    
    close(fig);  % Close the MOSSE figure before moving to next dataset
end

fclose(fid);

% Create combined figure for AUC graphs (one subplot per dataset)
numDatasets = length(successData);
figure;
for d = 1:numDatasets
    subplot(numDatasets, 1, d);
    plot(successData{d}.thresholds, successData{d}.successRates, '-o', 'LineWidth', 2);
    xlabel('Overlap Threshold');
    ylabel('Success Rate');
    title(sprintf('Success Plot for %s (AUC = %.4f)', successData{d}.dataset, successData{d}.auc));
    grid on;
end
% Save the combined AUC graph to a file
saveas(gcf, fullfile(pwd, 'AUC_graphs.png'));