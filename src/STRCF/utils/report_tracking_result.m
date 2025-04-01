function seq = report_tracking_result(seq, tracking_result)
% REPORT_TRACKING_RESULT - Stores the tracking result for the current frame.
%
% Stores the result (bounding box, center position, target size) and
% additional tracking metrics like peak score and lost status for the
% current frame into the sequence structure 'seq'.
% Handles different sequence formats (OTB, VOT).
%
% INPUT:
%   seq - The sequence structure, containing results from previous frames.
%   tracking_result - A structure containing results for the current frame,
%                     including fields like 'center_pos', 'target_size',
%                     'peak_score', and 'lost'.
%
% OUTPUT:
%   seq - The updated sequence structure.
%

% --- OTB Format ---
if strcmpi(seq.format, 'otb')
    % Calculate standard OTB bounding box [x, y, width, height]
    % Note: OTB uses 1-based indexing, pos is [row, col] (y, x)
    rect_position = round([tracking_result.center_pos([2,1]) - (tracking_result.target_size([2,1]) - 1)/2, tracking_result.target_size([2,1])]);

    % Initialize results structure on first frame
    if seq.frame == 1
        % Pre-allocate results arrays based on number of frames
        if isfield(seq, 'num_frames')
            num_frames = seq.num_frames;
        elseif isfield(seq, 'len') % Use 'len' if 'num_frames' doesn't exist
             num_frames = seq.len;
        else
            warning('Number of frames not found in seq structure, cannot pre-allocate results.');
            num_frames = seq.frame; % Allocate only for current frame if unknown
        end

        seq.results.res = NaN(num_frames, 4); % Bounding boxes
        seq.results.peak_scores = NaN(num_frames, 1); % Peak correlation scores
        seq.results.lost_status = false(num_frames, 1); % Lost status flag
    end

    % Store results for the current frame
    seq.results.res(seq.frame,:) = rect_position; % Store bounding box

    % Store peak score if provided
    if isfield(tracking_result, 'peak_score')
        seq.results.peak_scores(seq.frame) = tracking_result.peak_score;
    else
         % Assign NaN if not provided (e.g., first frame might not have it)
        seq.results.peak_scores(seq.frame) = NaN;
    end

    % Store lost status if provided
    if isfield(tracking_result, 'lost')
        seq.results.lost_status(seq.frame) = tracking_result.lost;
    else
         % Assign false if not provided
        seq.results.lost_status(seq.frame) = false;
    end

% --- VOT Format ---
elseif strcmpi(seq.format, 'vot')
    % For VOT, results are typically reported directly via the toolkit handle
    if seq.frame > 1
        bb_scale = 1; % VOT uses polygon format, this calculation might need adjustment
                      % depending on how center_pos and target_size relate to the polygon.
                      % Assuming axis-aligned bounding box for this example conversion.
        sz = tracking_result.target_size / bb_scale;
        tl = tracking_result.center_pos - (sz - 1)/2; % Top-left [row, col]
        br = tracking_result.center_pos + (sz - 1)/2; % Bottom-right [row, col]

        % Convert to VOT polygon format [x1, y1, x2, y1, x2, y2, x1, y2]
        x1 = tl(2); y1 = tl(1);
        x2 = br(2); y2 = br(1);
        result_polygon = round(double([x1 y1 x2 y1 x2 y2 x1 y2]));

        if any(isnan(result_polygon) | isinf(result_polygon))
            error('Illegal values in the VOT result polygon.')
        end

        % Report polygon via VOT handle
        seq.handle = seq.handle.report(seq.handle, result_polygon);

        % --- Optional: Storing extra info for VOT ---
        % If you need to store peak_score/lost status *alongside* VOT reporting,
        % you might need a separate mechanism or initialize seq.results similarly
        % to OTB, but be aware this isn't the standard VOT reporting method.
        % Example (uncomment and adapt if needed):
        % if seq.frame == 1 % Initialize here if needed for VOT custom storage
        %     num_frames = seq.num_frames; % Or seq.len
        %     seq.results.peak_scores = NaN(num_frames, 1);
        %     seq.results.lost_status = false(num_frames, 1);
        % end
        % if isfield(tracking_result, 'peak_score')
        %     seq.results.peak_scores(seq.frame) = tracking_result.peak_score;
        % end
        % if isfield(tracking_result, 'lost')
        %     seq.results.lost_status(seq.frame) = tracking_result.lost;
        % end
        % --- End Optional ---
    end
else
    error('Unknown sequence format');
end

end % function