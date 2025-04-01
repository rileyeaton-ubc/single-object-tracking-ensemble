function results = tracker(params)
% TRACKER - Core function for the STRCF tracker.
%
% Includes modifications to handle persistent lost status feedback:
% - Checks params.is_currently_lost (set by the caller).
% - Uses params.confidence_threshold to detect initial loss.
% - Uses params.redetection_threshold to detect re-acquisition.
% - Skips model updates if current_frame_lost is true.
% - Returns peak_score and lost status in results.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq'); % Keep params separate from seq info
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters (including new ones like confidence_threshold,
% redetection_threshold, is_currently_lost if not already set)
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;
temporal_regularization_factor = params.temporal_regularization_factor;
init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end
if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err %#ok<NASGU>
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % can be customized
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1); % Use the feature block with the largest spatial size

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y);
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);

    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));

    reg_window{i}(range_h, range_w) = params.reg_window_min;
end

% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);

% Allocate space for scores
scores_fs_feat = cell(1,1,num_feature_blocks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break; % Reached end of sequence
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1); % Convert to grayscale if needed
        end
    else
        seq.frame = 1; % Initialize frame counter
    end

    tic(); % Start frame timer

    % --- Initialize frame-specific status flags ---
    current_frame_lost = false; % Assume not lost initially for this frame
    current_frame_peak_score = NaN; % Initialize peak score for this frame

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;

        % --- Check if tracker was lost entering this frame ---
        was_lost = isfield(params, 'is_currently_lost') && params.is_currently_lost;
        if was_lost
            fprintf('--- Frame %d: Tracker starting in LOST state ---\n', seq.frame);
            % Optional: Implement search area widening here if desired.
            % This is complex due to interactions with feature/filter sizes.
            % Example (needs careful testing):
            % currentScaleFactor = currentScaleFactor * params.redetection_search_scale_factor;
            % Adjust scale limits if necessary:
            % currentScaleFactor = max(min_scale_factor, min(max_scale_factor, currentScaleFactor));
        end
        % --- End check ---

        % Translation search (usually runs only once, iter=1)
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            sample_scale = currentScaleFactor * scaleFactors; % Use current scale factor

            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);

            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);

            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);

            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};

            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz); % Resize to main block size
                scores_fs_sum = scores_fs_sum + scores_fs_feat{k};
            end

            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);

            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric'); % Spatial response map

            % --- Calculate Peak Score & Determine Lost Status ---
            peak_score = max(response(:));
            current_frame_peak_score = peak_score; % Store for reporting
            current_frame_lost = true; % Assume lost by default, prove otherwise

            if was_lost
                % Tracker was lost, check for re-acquisition
                if peak_score >= params.redetection_threshold
                    current_frame_lost = false; % Re-acquired!
                    fprintf('Frame %d: Target RE-ACQUIRED (Peak Score: %.4f >= Re-det Threshold: %.4f)\n', seq.frame, peak_score, params.redetection_threshold);
                else
                     % Still lost
                     fprintf('Frame %d: Target still lost (Peak Score: %.4f < Re-det Threshold: %.4f)\n', seq.frame, peak_score, params.redetection_threshold);
                     % current_frame_lost remains true
                end
            else
                % Tracker was NOT lost, check for normal loss
                if peak_score >= params.confidence_threshold
                     current_frame_lost = false; % Still tracking
                else
                    % Transitioning to lost state
                    fprintf('Frame %d: Target LOST (Peak Score: %.4f < Conf Threshold: %.4f)\n', seq.frame, peak_score, params.confidence_threshold);
                    % current_frame_lost remains true
                end
            end
            % --- End Lost Status Determination ---

            % Find peak location using Newton's method (even if confidence is low)
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);

            % Compute the translation vector in pixel-coordinates
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
            scale_change_factor = scaleFactors(sind);

            % Update position (potentially based on low confidence score if lost)
            old_pos = pos;
            pos = sample_pos + translation_vec;

            % Clamp position to image boundaries
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end

            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;

            % Adjust scale to prevent excessive changes
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end

            iter = iter + 1;
        end % End refinement loop (while iter <= ...)

    else % Handle first frame (seq.frame == 1)
        % No localization needed, use initial values
        current_frame_peak_score = 1.0; % Assign default high confidence
        current_frame_lost = false;    % Assume not lost on first frame
        was_lost = false; % Cannot be lost entering the first frame
        fprintf('--- Frame 1: Initializing Tracker ---\n');
    end % End localization step (if seq.frame > 1)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % --- Conditionally Update Model ---
    % Only update the model if the target is currently considered NOT lost
    % (i.e., tracking normally or just re-acquired in this frame)
    if ~current_frame_lost

        fprintf('Frame %d: Updating tracker model (Score: %.4f).\n', seq.frame, current_frame_peak_score);

        % Extract image region for training sample at the updated position/scale
        sample_pos = round(pos);
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);

        % Do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);

        % Compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);

        % Train the CF model for each feature block via ADMM
        for k = 1: numel(xlf)
            model_xf = xlf{k}; % Features for this block

            % Set temporal regularization factor (mu)
            if (seq.frame == 1) % No temporal regularization on first frame
                f_pre_f{k} = zeros(size(model_xf), 'like', model_xf); % Initialize previous filter
                mu = 0;
            else
                mu = temporal_regularization_factor(k);
            end

            % Initialize ADMM variables for this block
            f_f = single(zeros(size(model_xf), 'like', model_xf)); % Filter (f)
            g_f = f_f; % Auxiliary variable (g)
            h_f = f_f; % Lagrange multiplier (h)
            gamma  = init_penalty_factor(k); % Penalty factor (gamma)
            gamma_max = max_penalty_factor(k);
            gamma_scale_step = penalty_scale_step(k);

            % Use GPU if enabled
            if params.use_gpu
                model_xf = gpuArray(model_xf);
                f_f = gpuArray(f_f);
                f_pre_f{k} = gpuArray(f_pre_f{k});
                g_f = gpuArray(g_f);
                h_f = gpuArray(h_f);
                reg_window{k} = gpuArray(reg_window{k});
                yf{k} = gpuArray(yf{k});
            end

            % Pre-compute variables for ADMM speedup
            T = prod(filter_sz_cell{k}); % Use filter size T here
            S_xx = sum(conj(model_xf) .* model_xf, 3);
            Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);
            Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f); % Element-wise mult

            % Solve via ADMM algorithm
            iter_admm = 1;
            while (iter_admm <= admm_max_iterations)
                % Subproblem f (solve for filter)
                B = S_xx + T * (gamma + mu); % Denominator factor
                Sgx_f = sum(conj(model_xf) .* g_f, 3);
                Shx_f = sum(conj(model_xf) .* h_f, 3);

                % Update f_f (filter estimate)
                f_f = ((1/(T*(gamma + mu)) * bsxfun(@times, yf{k}, model_xf)) - ...
                       ((1/(gamma + mu)) * h_f) + (gamma/(gamma + mu)) * g_f) + ...
                       (mu/(gamma + mu)) * f_pre_f{k} - ...
                       bsxfun(@rdivide, ...
                              ( (1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .* yf{k})) ) + ...
                                ( (mu/(gamma + mu)) * Sfx_pre_f ) - ...
                                ( (1/(gamma + mu)) * (bsxfun(@times, model_xf, Shx_f)) ) + ...
                                ( (gamma/(gamma + mu)) * (bsxfun(@times, model_xf, Sgx_f)) ) ...
                              ), B);

                % Subproblem g (spatial regularization)
                g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f + h_f)), g_f));

                % Update h (Lagrange multiplier)
                h_f = h_f + (gamma * (f_f - g_f));

                % Update gamma (penalty factor)
                gamma = min(gamma_scale_step * gamma, gamma_max);

                iter_admm = iter_admm + 1;
            end % End ADMM iterations

            % Save the trained filter for the next frame
            f_pre_f{k} = f_f; % Filter used for temporal regularization in next frame
            cf_f{k} = f_f;    % Filter used for localization in next frame

        end % End loop over feature blocks (for k = ...)

    else % If current_frame_lost is true
         fprintf('Frame %d: Tracker is lost, SKIPPING model update.\n', seq.frame);
         % Keep the previous filter: f_pre_f already holds the last trained filter,
         % and cf_f will implicitly keep its value from the previous iteration
         % as the update loop is skipped. cf_f{k} remains unchanged.
    end
    % --- End Conditional Model Update ---


    % Update the target size (used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;

    % --- Prepare result structure for reporting ---
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    tracking_result.peak_score = current_frame_peak_score; % Use the determined score
    tracking_result.lost = current_frame_lost;           % Use the determined flag

    % Report results (update seq structure)
    seq = report_tracking_result(seq, tracking_result);

    % Update total time
    seq.time = seq.time + toc();

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end

        imagesc(im_to_show);
        hold on;

        % --- Adjust visualization based on lost status ---
        if current_frame_lost
            box_edge_color = 'r'; % Red if lost
            status_text = sprintf('Lost (Score: %.2f)', current_frame_peak_score);
        else
            box_edge_color = 'g'; % Green if tracking
            status_text = sprintf('Tracking (Score: %.2f)', current_frame_peak_score);
        end
        rectangle('Position', rect_position_vis, 'EdgeColor', box_edge_color, 'LineWidth', 2);
        text(10, 10, [int2str(seq.frame) '/' int2str(numel(seq.image_files))], 'color', [0 1 1], 'FontSize', 12);
        text(10, 25, status_text, 'color', box_edge_color, 'FontSize', 12, 'FontWeight', 'bold');
        % --- End status visualization adjustment ---

        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])

        drawnow;
    end % End visualization

end % End main loop (while true)

[~, results] = get_sequence_results(seq);

end % Function tracker
