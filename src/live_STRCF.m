% Function to serve as the live inference of STRCF model for a given set of frames
% MODIFIED to accept incoming lost status and return outgoing status/score
function [tempResultBox, currentLostStatus, currentPeakScore] = live_STRCF(resultBox, lastNFramePaths, previousLostStatus)
    % Setup path to STRCF
    strcfFolder = 'STRCF';
    addpath(strcfFolder); % Consider using setup_paths or ensuring path is set outside

    % --- Prepare sequence structure for run_STRCF ---
    imageStruct.format = "otb"; % Ensure this matches how results are handled
    imageStruct.len = numel(lastNFramePaths);
    imageStruct.init_rect = resultBox; % Initial bounding box for this batch
    imageStruct.s_frames = lastNFramePaths; % Cell array of image paths

    % --- Prepare parameters, including incoming lost status ---
    % Initialize params structure (or load defaults)
    params = []; % You might load default params here if run_STRCF doesn't
    % >>> IMPORTANT: Ensure run_STRCF loads default parameters if 'params' is empty or doesn't contain tracker params <<<

    params.is_currently_lost = previousLostStatus; % Pass the state IN

    % --- Run STRCF ---
    fprintf('Running STRCF (Previous Lost Status: %s)...\n', string(previousLostStatus));
    % Modify run_STRCF call if needed to accept separate params struct
    % Assuming run_STRCF uses the params within imageStruct or accepts it separately
    % Let's assume imageStruct is sufficient for run_STRCF as defined in STRCF repo
    imageStruct.params = params; % Add params to the seq structure if run_STRCF expects it there
    boxResult = run_STRCF(imageStruct); % run_STRCF returns the full results struct

    % --- Extract Results ---
    tempResultBox = []; % Initialize empty
    currentLostStatus = true; % Default to lost if results are bad
    currentPeakScore = NaN; % Default peak score

    % Extract bounding box for the last frame
    if isfield(boxResult, 'res') && ~isempty(boxResult.res) && size(boxResult.res, 1) > 0
        tempResultBox = boxResult.res(end-1,:);
        fprintf('Bounding Box for last frame: [%.2f, %.2f, %.2f, %.2f]\n', tempResultBox);
    else
        warning('Bounding box result field "res" not found or empty in STRCF result.');
    end

    % Extract lost status for the last frame
    if isfield(boxResult, 'lost_status') && ~isempty(boxResult.lost_status)
        currentLostStatus = boxResult.lost_status(end); % Get the boolean value
        fprintf('>> Tracker Lost Object: %s (Returned Status)\n', string(currentLostStatus));
    else
        fprintf('>> Lost status field "lost_status" not found in tracker results. Assuming LOST.\n');
        % currentLostStatus remains true (default)
    end

    % Extract peak score for the last frame
    if isfield(boxResult, 'peak_scores') && ~isempty(boxResult.peak_scores)
        currentPeakScore = boxResult.peak_scores(end); % Get score for the last frame
        fprintf('>> Peak Score: %.4f\n', currentPeakScore);
    else
         fprintf('>> Peak score field "peak_scores" not found in tracker results.\n');
         % currentPeakScore remains NaN (default)
    end

    fprintf('STRCF processing finished.\n');

    % Return the bounding box, the NEW lost status, and the peak score
    return

end % function