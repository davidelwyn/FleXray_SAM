%% Create a comparison GIF and MP4 from an existing FleXray sequence
rootFolder = '';     % Leave blank to choose a folder containing frame_####
outputParent = '';   % Leave blank to choose where to save the video/GIF
frameRate = 20;
writeGif = true;
writeMp4 = true;

if isempty(rootFolder)
    rootFolder = uigetdir(pwd, 'Choose FleXray sequence output');
    if isequal(rootFolder, 0), return; end
end
if isempty(outputParent)
    outputParent = uigetdir(pwd, 'Choose video output folder');
    if isequal(outputParent, 0), return; end
end
validateattributes(frameRate, {'numeric'}, {'scalar','positive','finite'});
folders = dir(fullfile(rootFolder, 'frame_*'));
ids = [];
paths = {};
for k = 1:numel(folders)
    token = regexp(folders(k).name, '^frame_(\d+)$', 'tokens', 'once');
    if folders(k).isdir && ~isempty(token)
        path = fullfile(rootFolder, folders(k).name, 'comparison.png');
        assert(isfile(path), 'Missing comparison image: %s', path);
        ids(end+1) = str2double(token{1}); %#ok<SAGROW>
        paths{end+1} = path; %#ok<SAGROW>
    end
end
assert(~isempty(ids), 'No comparison frames found.');
[~, order] = sort(ids);
paths = paths(order);
destination = fullfile(outputParent, ['comparison_' datestr(now, 'yyyymmdd_HHMMSS_FFF')]);
assert(~isfolder(destination), 'Output already exists.');
mkdir(destination);
if writeMp4
    video = VideoWriter(fullfile(destination, 'comparison.mp4'), 'MPEG-4');
    video.FrameRate = frameRate;
    open(video);
end
try
    for k = 1:numel(paths)
        img = imread(paths{k});
        if size(img,3) == 1, img = repmat(img,1,1,3); end
        if writeMp4, writeVideo(video, img); end
        if writeGif
            [indexed, map] = rgb2ind(img,256);
            gifPath = fullfile(destination, 'comparison.gif');
            if k == 1
                imwrite(indexed,map,gifPath,'gif','LoopCount',Inf,'DelayTime',1/frameRate);
            else
                imwrite(indexed,map,gifPath,'gif','WriteMode','append','DelayTime',1/frameRate);
            end
        end
    end
catch exception
    if writeMp4, close(video); end
    rethrow(exception);
end
if writeMp4, close(video); end
fprintf('Saved %d frames to %s\n', numel(paths), destination);
