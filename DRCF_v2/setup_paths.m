function setup_paths()

% Add the neccesary paths
pathstr= fileparts(mfilename('fullpath'));

% Tracker implementation
addpath(genpath([pathstr '/implementation/']));

% Utilities
addpath([pathstr '/utils/']);

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% Mtimesx
addpath([pathstr '/external_libs/mtimesx/']);

% mexResize
addpath([pathstr '/external_libs/mexResize/']);
