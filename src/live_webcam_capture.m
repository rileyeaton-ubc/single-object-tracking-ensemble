% -------------------------------------------------------------------------
% Script to use a USB webcam to run the ensemble model live.
% Each model uses its own seperate method to run on the data, and each 
% should be contained in their own script.
% -------------------------------------------------------------------------

% ---------------------------- CONFIGURATION ------------------------------
% Clear workspace and close existing figures
clearvars;
% close all;

% Variables for saving frame images
targetFolderName = 'live_frames';   % Name of the subfolder to save frame images to    
baseFileName = 'frame_';         % Base name for each saved frame image
fileExtension = 'png';           % Image format
fileIndex = 0;                   % Starting index of frame images
fullFolderPath = fullfile(pwd, targetFolderName);

% Sedtup path to STRCF
strcfFolder = 'STRCF';
addpath(strcfFolder);

% Desired video framerate (in best case scenario)
desiredFramerate = 10;

% Time (in seconds) for the model to run for, and the time it initially
% waits to start the tracking using the initial target area
delayTime = 5;
modelRunTime = 30;

% The percent of the middle of the webcam image that will be filled by the
% initial target area square
targetAreaPercentage = 0.15;

% Set the number of previous frames to pass to models (including current)
pastFrameCount = 2;

% Box colours
targetSquareColor = [91, 207, 244] / 255; % Initial target area square
detectedColor = 'g';
lostColor = 'r';

% ---------------------------- INITIALIZATION -----------------------------
% --- Webcam setup ---
% Make sure a webcam is available, and let user know if there are none
if isempty(webcamlist)
    error('No webcam found. Please connect a webcam try again.');
end
% Create the webcam object
cam = webcam();

% Check available webcam resolutions
% disp('Available resolutions:');
% disp(cam.AvailableResolutions);
% Set the webcam resolution
cam.Resolution = '640x480';

% --- File cleanup ---
% Check if frame saving folder exists
if isfolder(fullFolderPath)
    disp('Clearing all previous frame files...');

    % Store file deletion pattern and store all matching files
    deletePattern = fullfile(fullFolderPath, sprintf('%s*.%s', baseFileName, fileExtension));
    filesToDelete = dir(deletePattern);
    
    % If there are no files to delete, print a message
    if isempty(filesToDelete)
        disp('No files matching the pattern found to delete.');
    % If there are files to delete
    else
        % Delete the files matching the pattern
        try
            delete(deletePattern);
            disp('Successfully deleted previous image files matching the pattern.');
        % If there is an error in deleting the files, error out
        catch ME_delete
            error('Could not delete all files matching pattern. Error: %s', ME_delete.message);
        end
    end
else
    % Create the image folder if it doesnt exist
    disp('Image folder not found. Creating now...');
    mkdir(fullFolderPath);
end

% --- Display setup ---
% Capture one frame and store the resolution
frame = snapshot(cam);
[height, width, ~] = size(frame);

% Calculate the initial detection central square
scaleFactor = sqrt(targetAreaPercentage);
squareLength = scaleFactor * height;
squareX = (width - squareLength) / 2;
squareY = (height - squareLength) / 2;
resultBox = [squareX, squareY, squareLength, squareLength];

% Create a figure for displaying the live webcam feed and detection results
liveFig = figure('Name', 'SOT Ensemble Demo Live Feed - Close this window to stop', ...
              'Position', [100, 100, width, height], ...
              'NumberTitle', 'off');

% Display the first frame to set up the image object
liveImage = imshow(frame);

% --- Loop seeding variables ---
keepRunning = true;        % Flag to keep while loop running
frameNum = 0;              % Frame counter 
activeModelFrameCount = 0; % Counter for number of frames that the model runs
initialBox = true;         % Flag for when the initial target area is shown
currPredictionRect = [];   % The current predicted rectangle coordinates
frameFilePaths = [""];     % Array to store all frame image filepath strings
countdownSecondsPrinted = [delayTime+1]; % Array to store the seconds printed for countdown
persistent_lost_status = false; % Initialize: Assume tracker is not lost initially
latest_peak_score = NaN;   % Initialize peak score
totalLoopTimer = tic;      % Start timer before the loop
disp('The object within the blue target area will be tracked in: ')

% --------------------- MAIN CAPTURE AND PROCCESS LOOP --------------------
% Loop until figure is closed OR keepRunning is false
while keepRunning && ishandle(liveFig) 
    % Increment total frame number and start single frame timer
    frameNum = frameNum + 1; 
    frameTimer = tic;

    % Set the box color
    boxColor = detectedColor;
    
    % --- Image capture ---
    % Capture a single frame and save it as an image
    frame = snapshot(cam);
    outputFileName = sprintf('%s%d.%s', baseFileName, frameNum, fileExtension);
    currFrameFilePath = fullfile(fullFolderPath, outputFileName);
    try
        imwrite(frame, currFrameFilePath);
    % If this image save fails, quit the program
    catch ME_write
        keepRunning = false;
        error("Failed to save frame to '%s'.\nError message: %s", currFrameFilePath, ME_write.message);
    end
    % Store the current filename in the array of all frames
    frameFilePaths(end+1) = currFrameFilePath;
    
    %  If the initial target area is no longer present
    if not(initialBox)
        % Get the list of paths to the last N (pastFrameCount) frames
        lastNFramePaths = frameFilePaths(1,(length(frameFilePaths)-pastFrameCount):length(frameFilePaths));
        
        % Get the model ID to use from the helper function and switch on it
        modelID = determine_ensemble_submodel(resultBox);
        tempResultBox = [];
        switch modelID
            % MOSSE 
            case 1
                disp('Using MOSSE')
            % KCF
            case 2
                disp('Using KCF')
            % STRCF
            case 3 % STRCF
                % Pass the persistent status INTO live_STRCF
                [tempResultBox, persistent_lost_status, latest_peak_score] = live_STRCF(resultBox, lastNFramePaths, persistent_lost_status);
                % NOTE: We modified live_STRCF to return the new status and score
            
                % --- Use the persistent_lost_status to set box color ---
                if persistent_lost_status
                    boxColor = lostColor; % Use red if lost
                else
                    boxColor = detectedColor; % Use green if tracking/re-acquired
                end
                % --- Update resultBox if tracking ---
                % Only update the reference box if the tracker is NOT lost
                if ~persistent_lost_status && ~isempty(tempResultBox)
                     % Assuming tempResultBox holds the result for the *last* frame of the batch
                     % Check the dimensions/structure of tempResultBox as returned
                     if size(tempResultBox,1) >= 1
                         resultBox = tempResultBox(end,:);
                     else
                         warning('STRCF returned an empty or invalid result box.');
                         % Decide how to handle this - maybe mark as lost?
                         % persistent_lost_status = true;
                         % boxColor = lostColor;
                     end
                elseif isempty(tempResultBox)
                     warning('STRCF returned empty result.');
                     % persistent_lost_status = true; % Consider marking as lost if empty
                     % boxColor = lostColor;
                end % else: if lost, resultBox retains its previous value
            % C-COT
            case 4
                disp('Using C-COT')
            % DSST
            case 5
                disp('Using DSST')
            otherwise
                fprintf('Model ID %d is invalid\n', modelID);
        end
        
        % If the returned result box contains results, update the current box
        if not(isempty(tempResultBox))
            resultBox = tempResultBox;
        end

        % Add one frame to the active model frame counter, and check the
        % time elapsed since the model has started running
        activeModelFrameCount = activeModelFrameCount + 1;
        activeModelTime = toc(totalLoopTimer) - delayTime;
        % If the model has been running longer than desired, quit the loop
        % and print the average fps
        if activeModelTime > modelRunTime
            keepRunning = false;
            disp('The model has now completed its runtime. Average fps:')
            disp(activeModelFrameCount/activeModelTime);
        end

    % If the initial target area is still present
    else
        % Store the total time
        totalTime = toc(totalLoopTimer);
        % If the total time has exceeded the delay, start the main model
        % loop on the next iteration
        if totalTime > delayTime
            initialBox = false;
            totalTimer = tic;
        else
            % Print the number of seconds until the target area captures
            % the target object to track (if not already printed)
            countdownTime = ceil(delayTime-totalTime);
            if not(ismember(countdownTime,countdownSecondsPrinted))
                countdownSecondsPrinted(end+1) = countdownTime;
                fprintf("%d\n", countdownTime);
            end
            
            % Set the box to blue
            boxColor = targetSquareColor;
        end
    end
    
    % --- Display results ---
    % If the figure is still alive and the image to display is valid,
    % update them to show the new frame that is bieng predicted
    if ishandle(liveFig) && isvalid(liveImage)
        set(liveImage, 'CData', frame);
    % If figure was closed
    else
        % Stop the loop
        keepRunning = false;
        disp('Figure window closed by user.');
    end

    % Remove the previous rectangle (prediction bounds)
    delete(currPredictionRect);

    % Get the axes where the image is displayed
    currentAxes = get(liveImage, 'Parent');
    % Turn on hold for the axes so the rectangle adds to the image
    hold(currentAxes, 'on');
    % Draw the rectangle and store its value
    currPredictionRect = rectangle(currentAxes, ...
                      'Position', resultBox, ...
                      'EdgeColor', boxColor, ... 
                      'LineWidth', 1);

    % Release the axes hold
    hold(currentAxes, 'off');
    
    % Force the figure to update
    drawnow;

    % --- Calculate required pause ---
    targetFrameTime = 1/desiredFramerate;
    % Store the time taken for all previous steps, and calculate how long
    % we need to pause for
    elapsedTime = toc(frameTimer); 
    pauseDuration = targetFrameTime - elapsedTime;

    % If processing was faster than desiredFramerate add a pause to prevent
    % excessive CPU usage
    if pauseDuration > 0
        % disp('pausing');
        pause(pauseDuration);
    else
        % prevent matlab from becoming completely unresponsive
        pause(0.001);
    end
end

% ------------------------------- CLEANUP ---------------------------------
% Release the webcam
disp('Stopping webcam...');
clear cam; 

% Ensure the figure is closed
if ishandle(liveFig)
    close(liveFig); 
end

% Remove subfolder from path
rmpath(strcfFolder);

% Print end message
disp('Webcam stopped and resources released.');

% ---------------------------- HELPER FUNCTIONS ---------------------------
% --- Determine which submodel to use based on current parameters ---
function modelID = determine_ensemble_submodel(resultBox)
    % MODEL ID LEGEND:
    % 1 - MOSSE
    % 2 - KCF
    % 3 - STRCF
    % 4 - C-COT
    % 5 - DSST
    % disp(resultBox);
    modelID = 3;
    return
end