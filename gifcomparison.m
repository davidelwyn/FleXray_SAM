%% Create GIF from FleXray comparison images

rootFolder = 'C:\Users\sce9dw1\Github\FleXray_SAM\outputs\biplane\LevelTM_sequence_0020_0060';
outputGif  = fullfile(rootFolder, 'comparison_animation.gif');

% Frame range
firstFrame = 20;
lastFrame  = 60;

% GIF timing
delayTime = 0.10;  % seconds per frame (0.10 = 10 fps)

for frameNum = firstFrame:lastFrame

    % Build path to comparison.png
    imageFile = fullfile(rootFolder, ...
        sprintf('frame_%04d', frameNum), ...
        'comparison.png');

    if ~isfile(imageFile)
        warning('Missing: %s', imageFile);
        continue
    end

    % Read PNG
    img = imread(imageFile);

    % Convert RGB image to indexed image for GIF
    [imgIndexed, colourMap] = rgb2ind(img, 256);

    % Write first frame, then append subsequent frames
    if frameNum == firstFrame
        imwrite(imgIndexed, colourMap, outputGif, 'gif', ...
            'LoopCount', Inf, ...
            'DelayTime', delayTime);
    else
        imwrite(imgIndexed, colourMap, outputGif, 'gif', ...
            'WriteMode', 'append', ...
            'DelayTime', delayTime);
    end
end

fprintf('GIF saved to:\n%s\n', outputGif);

%%
%% Create MP4 from FleXray comparison images

rootFolder = 'C:\Users\sce9dw1\Github\FleXray_SAM\outputs\biplane\LevelTM_sequence_0020_0060';
outputVideo = fullfile(rootFolder, 'comparison_video.mp4');

% Frame range
firstFrame = 20;
lastFrame  = 60;

% Playback speed
frameRate = 10;  % frames per second

% Create video
v = VideoWriter(outputVideo, 'MPEG-4');
v.FrameRate = frameRate;
v.Quality = 95;

open(v);

for frameNum = firstFrame:lastFrame

    imageFile = fullfile(rootFolder, ...
        sprintf('frame_%04d', frameNum), ...
        'comparison.png');

    if ~isfile(imageFile)
        warning('Missing: %s', imageFile);
        continue
    end

    img = imread(imageFile);

    % VideoWriter expects RGB
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end

    writeVideo(v, img);
end

close(v);

fprintf('Video saved to:\n%s\n', outputVideo);