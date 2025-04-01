% Example of running a script in a subfolder
fprintf('\n------ Start of Subfolder Script Run ------\n');

% Set variables
subFolder = 'STRCF';
runScript = 'demo_STRCF.m';
% Construct the relative path to the target script
scriptPath = fullfile(subFolder, runScript);
% We need to use the 'run' command with the path inside evalc
runScript = sprintf('run(''%s'')', scriptPath);

% Initialize required output, error, and exception variables
stdout_capture = ''; 
stderr_capture = '';
execution_error = [];

% Execute script and capture output and errors
disp(['Running script: ', scriptPath]);
try
    % evalc runs the command and captures its Command Window output
    stdout_capture = evalc(runScript);
    disp('Execution completed successfully.');
catch ME % Catch any error that occurred during the execution
    disp('Execution failed with an error.');
    % ME is an MException object containing error details
    stderr_capture = ME.message; % Get the primary error message
    % You can also get the full stack trace if needed:
    % stderr_capture = getReport(ME, 'extended', 'hyperlinks','off');
    execution_error = ME; % Store the full exception object
end

% Display the results that were captured
fprintf('\n--- Captured STDOUT ---\n');
if ~isempty(stdout_capture)
    fprintf('%s\n', stdout_capture);
else
    fprintf('(No stdout captured)\n');
end

fprintf('\n--- Captured STDERR ---\n');
if ~isempty(stderr_capture)
    fprintf('Error Message: %s\n', stderr_capture);
    % Display more error details (the strack trace)
    if ~isempty(execution_error)
        disp('Full Error Details:');
        disp(execution_error);
        disp('Stack Trace:');
        disp(execution_error.stack);
    end
else
    fprintf('(No stderr captured)\n');
end

fprintf('\n------ End of Subfolder Script Run ------\n');