function [seq, results] = get_sequence_results(seq)

if strcmpi(seq.format, 'otb')
    results.type = 'rect';
    results.res = seq.results.res;

    % Copy peak scores if they exist
    if isfield(seq.results, 'peak_scores')
        results.peak_scores = seq.results.peak_scores;
    end

    % Copy lost status if it exists
    if isfield(seq.results, 'lost_status')
        results.lost_status = seq.results.lost_status;
    end
elseif strcmpi(seq.format, 'vot')
    seq.handle.quit(seq.handle);
else
    error('Uknown sequence format');
end

if isfield(seq, 'time')
    results.fps = seq.num_frames / seq.time;
else
    results.fps = NaN;
end