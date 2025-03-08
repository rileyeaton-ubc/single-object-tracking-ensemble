function [images, region] = vot_tracker_initialize();

% read the images file
%{
fid = fopen('images.txt','r'); 
images = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
images = images{1};

%}

% Define the path to the image sequence
datadir = '../data/';
dataset = 'OTB2015/Bird1';
path = fullfile(datadir, dataset);
img_path = fullfile(path, 'img')
D = dir(fullfile(img_path, '*.jpg'))
D = D(~[D.isdir]);
seq_len = length(D);

if seq_len == 0
    error('No image files found in the directory');
end

img_files = sort({D.name});

images = fullfile(img_path, img_files);

first_image_path = fullfile(img_path, img_files{1});
first_image = imread(first_image_path);

figure;
imshow(first_image);
title('Select the Basketball and Press Enter');

h = drawrectangle;
wait(h);

region = round(h.Position);

%{
region_file = fullfile(datadir, dataset, 'region.txt');
dlmwrite(region_file, region, 'delimiter', ' ');

disp(['Bounding box saved to region.txt: ', num2str(region)]);

if exist([img_path num2str(1, '%04i.jpg')], 'file'),
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']);
else
    error('No image files found in the directory.');
end

% read the region
region = dlmread('region.txt');

if numel(region) == 4
	return;
end;

if numel(region) >= 6 && mod(numel(region), 2) == 0
	return;
end;

error('Illegal format of the input region!');
%}

end
