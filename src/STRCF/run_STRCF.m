function results = run_STRCF(seq, res_path, bSaveImage)
% RUN_STRCF - Sets up parameters and runs the STRCF tracker.
%
% This version is modified to accept an incoming lost status via the
% 'seq.params.is_currently_lost' field and pass it to the tracker.
%
% INPUT:
%   seq        - The sequence structure, potentially containing:
%                - seq.params.is_currently_lost: Boolean indicating if the
%                  tracker was lost in the previous state.
%                - Standard sequence fields (s_frames, init_rect, etc.)
%   res_path   - (Optional) Path for saving results (used in original STRCF).
%   bSaveImage - (Optional) Flag for saving images (used in original STRCF).
%
% OUTPUT:
%   results    - The results structure returned by the tracker, including
%                bounding boxes, peak scores, and lost status.
%

% --- Default Tracker Parameters (Hardcoded in this runner script) ---
% Note: It's generally better practice to define these in a separate
% function or pass them in, but we follow the original structure here.

% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 10;
hog_params.nDim = 31;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;

% Which features to include
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};

% Global feature parameters
params.t_global.cell_size = 4;                  % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 5;           % The scaling of the target size to get the search area (Adjusted from 10, 5 is common)
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Spatial regularization window_parameters
params.feature_downsample_ratio = [4]; % Feature downsample ratio (adjust if features change)
params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;          % The minimum value of the regularization window

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma
params.temporal_regularization_factor = [15 15]; % The temporal regularization parameters (Adjust per feature if needed)

% ADMM parameters
params.max_iterations = [2 2];          % Max ADMM iterations per feature
params.init_penalty_factor = [1 1];     % Initial penalty factor per feature
params.max_penalty_factor = [0.1, 0.1]; % Max penalty factor per feature
params.penalty_scale_step = [10, 10];   % Penalty scale step per feature

% Scale parameters for the translation model
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor

% Visualization
params.visualization = 0;               % Visualize tracking and detection scores (Set to 1 for debugging)

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default


% --- Merge incoming state from seq structure if it exists ---
% This allows the caller (e.g., live_STRCF) to pass the persistent lost status
if isfield(seq, 'params') && isfield(seq.params, 'is_currently_lost')
     params.is_currently_lost = seq.params.is_currently_lost;
     fprintf('[run_STRCF] Received incoming lost status: %s\n', string(params.is_currently_lost));
else
     % If not provided by caller, tracker will use default from init_default_params
     fprintf('[run_STRCF] Incoming lost status not provided in seq.params. Tracker will use default.\n');
     % params.is_currently_lost will be set by init_default_params inside tracker.m
end
% --- End Merge ---


% --- Initialize ---
params.seq = seq; % Pass the sequence information into the params struct for the tracker


% --- Run tracker ---
% The 'tracker' function will handle calling 'init_default_params' to ensure
% all necessary parameters (including the new loss/redetection ones) are set,
% respecting any values already present in the 'params' struct passed here.
results = tracker(params);

end % function
