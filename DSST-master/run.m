function run()
    config = fileread("../config.json"); 
    config = jsondecode(config);
    datapath = config.datafile_path;%Read the config file to access datasets

    datasets = dir(datapath);
    datasets = datasets(~ismember({datasets.name},{'.', '..'}));

    for k=1:length(datasets)
        data = fullfile(datapath, datasets(k).name);
        wrapper(data);
        %Code needs to be updated so it automatically reads the first frame
        %data from ground truth.

        %Code needs to be updated to handle the output of the result.
    end
end
