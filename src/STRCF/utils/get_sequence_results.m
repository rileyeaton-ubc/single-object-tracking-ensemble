function [seq, results] = get_sequence_results(seq)
% GET_SEQUENCE_RESULTS - Finalizes and returns the tracking results.
%
% Packages the results accumulated in the 'seq' structure into the final
% 'results' structure, suitable for evaluation. Calculates FPS.
% Includes standard results (bounding boxes) and additional metrics
% like peak scores and lost status if available (primarily for OTB format).
%
% INPUT:
%   seq - The sequence structure containing accumulated results.
%
% OUTPUT:
%   seq     - The input sequence structure (potentially updated by VOT quit).
%   results - A structure containing the final tracking results, including
%             'res' (bounding boxes), 'fps', and potentially 'peak_scores'
%             and 'lost_status'.
%

% --- OTB Format ---
if strcmpi(seq.format, 'otb')
    results.type = 'rect'; % Standard OTB result type

    % Check if results were stored and copy them
    if isfield(seq, 'results') && isfield(seq.results, 'res')
        results.res = seq.results.res;
    else
        warning('No bounding box results found in seq.results.res for OTB format.');
        results.res = []; % Return empty if not found
    end

    % Copy peak scores if they exist
    if isfield(seq, 'results') && isfield(seq.results, 'peak_scores')
        results.peak_scores = seq.results.peak_scores;
    else
        % Optionally return NaNs or empty if not found
        results.peak_scores = NaN(size(results.res, 1), 1);
    end

    % Copy lost status if it exists
    if isfield(seq, 'results') && isfield(seq.results, 'lost_status')
        results.lost_status = seq.results.lost_status;
    else
        % Optionally return false or empty if not found
         results.lost_status = false(size(results.res, 1), 1);
    end

% --- VOT Format ---
elseif strcmpi(seq.format, 'vot')
    % For VOT, the primary result reporting happens via the handle.
    % This function mainly handles quitting the toolkit interaction.
    % No standard 'res' field is typically returned here for VOT evaluation,
    % as evaluation is handled by the VOT toolkit based on reported polygons.
    seq.handle.quit(seq.handle);
    results.type = 'vot'; % Indicate VOT format
    results.res = [];     % Standard VOT doesn't return BBs here

    % --- Optional: Returning extra info for VOT ---
    % If you implemented custom storage for VOT in report_tracking_result,
    % you could retrieve it here. Uncomment and adapt if needed.
    % if isfield(seq, 'results') && isfield(seq.results, 'peak_scores')
    %     results.peak_scores = seq.results.peak_scores;
    % end
    % if isfield(seq, 'results') && isfield(seq.results, 'lost_status')
    %     results.lost_status = seq.results.lost_status;
    % end
    % --- End Optional ---

else
    error('Unknown sequence format');
end

% Calculate FPS
if isfield(seq, 'time') && isfield(seq, 'num_frames') && seq.time > 0
    results.fps = seq.num_frames / seq.time;
elseif isfield(seq, 'time') && isfield(seq, 'len') && seq.time > 0 % Fallback to 'len'
     results.fps = seq.len / seq.time;
else
    results.fps = NaN; % Assign NaN if time or frame count is missing/invalid
end

end % function