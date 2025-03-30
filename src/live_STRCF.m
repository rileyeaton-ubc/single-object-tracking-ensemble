% Run the standard STRCF model live
disp('\n------ Start of Live STRCF ------\n');

% Set subfolder name and add it to paths
strcfFolder = 'STRCF';
addpath(strcfFolder);
setup_paths();

% Set remaining variables
sequencesFolder = 'sequences';
sequencePath = strcat(strcfFolder, '/', sequencesFolder);
video = 'Human3';
videoPath = [sequencePath '/' video];

% store the frame sequence using load_video_info
[seq] = load_video_info(videoPath);

% Run STRCF
results = run_STRCF(seq);
%results = run_DeepSTRCF(seq);

% Store boxes from results
pd_boxes = results.res;

% Remove subfolder from path
rmpath(strcfFolder); 

disp('\n------ End of Live STRCF ------\n');
