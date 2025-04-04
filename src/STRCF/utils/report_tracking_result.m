function seq = report_tracking_result(seq, result)
if strcmpi(seq.format, 'otb')
    rect_position = round([result.center_pos([2,1]) - (result.target_size([2,1]) - 1)/2, result.target_size([2,1])]);
    % Initialize results structure on first frame
    if seq.frame == 1
        if isfield(seq, 'num_frames')
            num_frames = seq.num_frames;
        else
            % Set only for current frame if unknown
            warning('Number of frames not found in seq structure');
            num_frames = seq.frame;
        end

        % Set bounding boxes, peak correlation scores, and lost status
        seq.results.res = NaN(num_frames, 4);
        seq.results.peak_scores = NaN(num_frames, 1);
        seq.results.lost_status = false(num_frames, 1);
    end

    % Store results for the current frame
    seq.results.res(seq.frame,:) = rect_position;

    % Store peak score if provided
    if isfield(result, 'peak_score')
        seq.results.peak_scores(seq.frame) = result.peak_score;
    else
        % Otherwise store NaN (first frame)
        seq.results.peak_scores(seq.frame) = NaN;
    end

    % Store lost status if provided
    if isfield(result, 'lost')
        seq.results.lost_status(seq.frame) = result.lost;
    else
        % Otherwise store false
        seq.results.lost_status(seq.frame) = false;
    end
elseif strcmpi(seq.format, 'vot')
    if seq.frame > 1
        bb_scale = 1;
        sz = result.target_size / bb_scale;
        tl = result.center_pos - (sz - 1)/2;
        br = result.center_pos + (sz - 1)/2;
        x1 = tl(2); y1 = tl(1);
        x2 = br(2); y2 = br(1);
        result_box = round(double([x1 y1 x2 y1 x2 y2 x1 y2]));
        if any(isnan(result_box) | isinf(result_box))
            error('Illegal values in the result.')
        end
%         if any(result_box < 0)
%             error('Negative values')
%         end
        seq.handle = seq.handle.report(seq.handle, result_box);
    end
else
    error('Unknown sequence format');
end