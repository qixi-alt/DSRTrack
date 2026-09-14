clear;
clc;

%% Settings: only edit this section
sourceDir = '/Users/yutongli/Downloads/VTUAV-ST-results';
outputDir = '/Users/yutongli/Projects/RGBT_Tracking/RGT_ToolKit/toolkit_VTUAV/BBresults';

trackerName = 'DSRTrack';
sourcePrefix = '';
expectedCount = 176;

%% Copy and rename
% Example: dimp50_rgbt_5_animal_001.txt -> QAT_animal_001.txt

files = dir(fullfile(sourceDir, [sourcePrefix, '*.txt']));
if isempty(files)
    error('No result files matching "%s*.txt" were found in:\n%s', ...
        sourcePrefix, sourceDir);
end

if numel(files) ~= expectedCount
    warning('Expected %d result files, but found %d.', ...
        expectedCount, numel(files));
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

outputNames = cell(numel(files), 1);
for i = 1:numel(files)
    sourceName = files(i).name;
    sequenceName = sourceName(length(sourcePrefix) + 1:end - 4);
    outputName = sprintf('%s_%s.txt', trackerName, sequenceName);

    sourcePath = fullfile(sourceDir, sourceName);
    outputPath = fullfile(outputDir, outputName);
    copyfile(sourcePath, outputPath, 'f');
    outputNames{i} = outputName;
end

if numel(unique(outputNames)) ~= numel(outputNames)
    error('Duplicate output filenames were generated.');
end

generatedFiles = dir(fullfile(outputDir, [trackerName, '_*.txt']));
fprintf('Generated %d files in:\n%s\n', numel(generatedFiles), outputDir);
fprintf('Example: %s -> %s\n', files(1).name, outputNames{1});
