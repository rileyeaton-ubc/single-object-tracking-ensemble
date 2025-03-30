% Function to serve as the live inference of STRCF model for a given set of
% frames
function tempResultBox = live_STRCF(resultBox, lastNFramePaths)

% Sedtup path to STRCF
strcfFolder = 'STRCF';
addpath(strcfFolder);

% Update the image structure to pass to the model
imageStruct.format = "otb";
imageStruct.len = 1;
imageStruct.init_rect = resultBox;
imageStruct.s_frames = lastNFramePaths;

% Run STRCF and return the results
boxResult = run_STRCF(imageStruct);
tempResultBox = boxResult.res;
return