%% Set folders here, or leave blank to use the selection dialogs
rootFolder = '';                  % Data root containing SAM/, or SAM folder itself
trialConfig = '';                 % Optional full path to a trial .cfg
maskFolder = '';                  % Existing Python sequence output (frame_#### folders)
outputParent = '';                % A new export subfolder is created here
dilationPixels = 30;               % Disk radius in original pixels; 0 = no dilation
exportNames = {'femur', 'tibia', 'combined'};
exportLabels = {{'femurs'}, {'tibiae'}, {'femurs', 'tibiae'}};
% Each export gets separate masked X-rays, binary masks, and a SAM cfg.
% The same dilation radius is applied independently to each bone mask.
frameNumbers = [];                % [] = all available mask frames; or e.g. 20:60
backgroundValue = 0;              % Black background in the original intensity scale
% Empty masks export as background-only frames and are listed in frame_status.csv.
% Keep these frames to preserve camera alignment; they contain no tracking evidence.
showPreview = true;
%% Choose the activity/trial and existing segmentation results
if isempty(trialConfig)
    if isempty(rootFolder)
        rootFolder = uigetdir(pwd, 'Choose data root or SAM folder');
        if isequal(rootFolder, 0), return; end
    end
    samFolder = rootFolder;
    if isfolder(fullfile(rootFolder, 'SAM')), samFolder = fullfile(rootFolder, 'SAM'); end
    configs = dir(fullfile(samFolder, '*.cfg'));
    assert(~isempty(configs), 'No trial .cfg files in %s', samFolder);
    [selection, ok] = listdlg('ListString', {configs.name}, 'SelectionMode', 'single', ...
        'PromptString', 'Choose activity / trial', 'ListSize', [600 280]);
    if ~ok, return; end
    trialConfig = fullfile(samFolder, configs(selection).name);
end
if isempty(maskFolder)
    maskFolder = uigetdir(pwd, 'Choose matching FleXray sequence output');
    if isequal(maskFolder, 0), return; end
end
if isempty(outputParent)
    outputParent = uigetdir(pwd, 'Choose output folder');
    if isequal(outputParent, 0), return; end
end
outputFolder = fullfile(outputParent, ['masked_preview_' datestr(now, 'yyyymmdd_HHMMSS_FFF')]);

% This exports a visual masked trial. Constant background is NOT an optimizer
% exclusion mask. SAM must support an ROI/weight mask to truly ignore pixels.
% Source images/configuration are never overwritten. Requires Image Processing Toolbox.
%% Validate all selected frames before exporting
validateattributes(dilationPixels, {'numeric'}, {'scalar','integer','nonnegative','finite'});
validateattributes(backgroundValue, {'numeric'}, {'scalar','nonnegative','finite'});
assert(numel(exportNames) == numel(exportLabels) && ~isempty(exportNames), 'Export names and labels must match.');
assert(numel(unique(exportNames)) == numel(exportNames), 'Export names must be unique.');
for e = 1:numel(exportNames)
    assert(~isempty(regexp(exportNames{e}, '^[A-Za-z0-9_-]+$', 'once')), 'Use simple export folder names.');
    assert(~isempty(exportLabels{e}), 'Choose at least one mask label per export.');
end
assert(~isfolder(outputFolder) && ~isfile(outputFolder), 'Output already exists: %s', outputFolder);
configText = fileread(trialConfig);
configDir = fileparts(trialConfig);
cameraTokens = regexp(configText, '(?m)^CameraRootDir\s+([^\r\n]+)', 'tokens');
assert(numel(cameraTokens) == 2, 'Expected exactly two CameraRootDir entries.');
cameraNames = cell(1,2);
for camera = 1:2
    p = strrep(strtrim(cameraTokens{camera}{1}), '\', '/');
    [~, cameraNames{camera}] = fileparts(p);
end
folders = dir(fullfile(maskFolder, 'frame_*'));
ids = [];
for k = 1:numel(folders)
    token = regexp(folders(k).name, '^frame_(\d+)$', 'tokens', 'once');
    if folders(k).isdir && ~isempty(token), ids(end+1) = str2double(token{1}); end %#ok<SAGROW>
end
ids = sort(ids);
if ~isempty(frameNumbers)
    validateattributes(frameNumbers, {'numeric'}, {'vector','integer','positive','finite'});
    assert(all(ismember(frameNumbers, ids)), 'Some requested frames have no masks.');
    ids = sort(unique(frameNumbers));
end
assert(~isempty(ids), 'No mask frames found in %s', maskFolder);
assert(all(diff(ids) == 1), 'Use consecutive frames so SAM frame order remains meaningful.');
fprintf('Found %d mask frame pairs (source frames %d to %d).\n', numel(ids), ids(1), ids(end));
sources = cell(numel(ids),2);
names = cell(numel(ids),2);
emptyMasks = false(numel(ids), 2, numel(exportNames));
for k = 1:numel(ids)
    frameDir = fullfile(maskFolder, sprintf('frame_%04d', ids(k)));
    report = jsondecode(fileread(fullfile(frameDir, 'report.json')));
    assert(strcmp(report.coordinates, 'original pixels; origin top-left; boxes [x0,y0,x1,y1], end exclusive'), ...
        'Masks must be in original image coordinates.');
    for camera = 1:2
        source = report.views.(sprintf('view%d', camera)).input;
        [parent, name, ext] = fileparts(source);
        [~, sourceTrial] = fileparts(parent);
        assert(strcmp(sourceTrial, cameraNames{camera}), 'Trial mismatch: %s versus %s', sourceTrial, cameraNames{camera});
        token = regexp([name ext], '\.(\d+)\.tiff?$', 'tokens', 'once');
        assert(~isempty(token) && str2double(token{1}) == ids(k), 'Source frame number mismatch.');
        img = imread(source);
        assert(ismatrix(img) && (isa(img,'uint8') || isa(img,'uint16')), 'Expected grayscale uint8/uint16 TIFF.');
        assert(backgroundValue <= double(intmax(class(img))) && backgroundValue == fix(backgroundValue), 'Background outside image intensity range.');
        for e = 1:numel(exportNames)
            selectedMask = readTrialMask(frameDir, camera, exportLabels{e}, size(img));
            emptyMasks(k,camera,e) = ~any(selectedMask(:));
        end
        sources{k,camera} = source;
        names{k,camera} = [name ext];
    end
end
%% Export full-sized TIFFs, binary masks, and a copied SAM configuration
mkdir(outputFolder);
for e = 1:numel(exportNames)
variantFolder = fullfile(outputFolder, exportNames{e});
mkdir(variantFolder);
maskLabels = exportLabels{e};
emptyForExport = emptyMasks(:,:,e);
% MATLAB stores columns consecutively: all C1 rows, then all C2 rows.
frameStatus = table(repmat(ids(:), 2, 1), ...
    [ones(numel(ids),1); 2*ones(numel(ids),1)], emptyForExport(:), ...
    'VariableNames', {'source_frame', 'camera', 'empty_mask'});
writetable(frameStatus, fullfile(variantFolder, 'frame_status.csv'));
if any(emptyForExport(:))
    warning('SAMPreview:EmptyMask', ...
        '%s: %d empty masks will be exported as background-only images. See frame_status.csv.', ...
        exportNames{e}, nnz(emptyForExport));
end
cameraDirs = cell(1,2);
for camera = 1:2
    cameraDirs{camera} = fullfile(variantFolder, cameraNames{camera});
    mkdir(cameraDirs{camera});
    mkdir(fullfile(variantFolder, sprintf('masks_C%d', camera)));
end
for k = 1:numel(ids)
    frameDir = fullfile(maskFolder, sprintf('frame_%04d', ids(k)));
    for camera = 1:2
        img = imread(sources{k,camera});
        mask = readTrialMask(frameDir, camera, maskLabels, size(img));
        if dilationPixels > 0, mask = imdilate(mask, strel('disk', dilationPixels, 0)); end
        masked = img;
        masked(~mask) = cast(backgroundValue, 'like', img);
        imwrite(masked, fullfile(cameraDirs{camera}, names{k,camera}), 'Compression', 'none');
        imwrite(uint8(mask)*255, fullfile(variantFolder, sprintf('masks_C%d', camera), sprintf('mask_%04d.png', ids(k))));
        if showPreview && k == 1
            figure('Name', sprintf('SAM %s preview C%d, source frame %d', exportNames{e}, camera, ids(k)));
            subplot(1,2,1); imshow(img); title('Original');
            subplot(1,2,2); imshow(masked); title(sprintf('Masked, dilation %d px', dilationPixels));
        end
    end
end
% Make resource paths absolute because the copied cfg lives in a new folder.
lines = regexp(configText, '\r\n|\n|\r', 'split');
camera = 0;
for k = 1:numel(lines)
    token = regexp(lines{k}, '^(CameraRootDir|mayaCam_csv|VolumeFile|MeshFile)\s+(.+)$', 'tokens', 'once');
    if isempty(token), continue; end
    if strcmp(token{1}, 'CameraRootDir')
        camera = camera + 1;
        p = cameraDirs{camera};
    else
        p = strtrim(token{2});
        if isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/])', 'once')), p = fullfile(configDir,p); end
    end
    lines{k} = [token{1} ' ' strrep(p, '\', '/')];
end
[~, configName] = fileparts(trialConfig);
% Retain the original model entries in each cfg; only the X-ray input changes.
outputConfig = fullfile(variantFolder, [configName '_' exportNames{e} '_masked.cfg']);
fid = fopen(outputConfig, 'wt');
assert(fid >= 0, 'Cannot write configuration.');
fprintf(fid, '%s\n', lines{:});
fclose(fid);
save(fullfile(variantFolder, 'export_settings.mat'), 'trialConfig', 'maskFolder', 'dilationPixels', ...
    'maskLabels', 'backgroundValue', 'ids', 'sources', 'emptyForExport');
fprintf('Exported %d frame pairs (source frames %d to %d).\nOpen in SAM:\n%s\n', numel(ids), ids(1), ids(end), outputConfig);
end
fprintf('If exporting a subset, check SAM frame numbering before using existing tracking.\n');

function mask = readTrialMask(frameDir, camera, labels, imageSize)
mask = false(imageSize);
for j = 1:numel(labels)
    path = fullfile(frameDir, sprintf('view%d', camera), [labels{j} '_mask.png']);
    labelMask = imread(path);
    assert(isequal(size(labelMask), imageSize), 'Mask dimensions differ from X-ray: %s', path);
    mask = mask | labelMask > 0;
end
% An existing all-zero mask is valid: dilation keeps it empty and export
% produces a background-only image. Missing files/dimension errors still stop.
end
