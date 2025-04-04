% Function to serve as the live inference of STRCF model for a given set of frames
function [tempResultBox, currentLostStatus, currentPeakScore] = live_STRCF(resultBox, lastNFramePaths, previousLostStatus)

% Setup path to STRCF
strcfFolder = 'STRCF';
addpath(strcfFolder);

% Update the image structure to pass to the model
imageStruct.format = "otb";
imageStruct.len = numel(lastNFramePaths);
imageStruct.init_rect = resultBox;
imageStruct.s_frames = lastNFramePaths;

% Initialize params, and pass in previous lost status
params = [];
params.is_currently_lost = previousLostStatus; 

% Run STRCF and pass params
imageStruct.params = params; 
boxResult = run_STRCF(imageStruct);

% Initialize result variables
tempResultBox = [];
currentLostStatus = true;
currentPeakScore = NaN;

% Get resulting bounding box 
tempResultBox = boxResult.res(end-1,:);

% Get the lost status
currentLostStatus = boxResult.lost_status(end);

% Get peak score for the last frame
currentPeakScore = boxResult.peak_scores(end);

% Return the bounding box, lost status, and peak score
return