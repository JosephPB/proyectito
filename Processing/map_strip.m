%% Library paths 
% User note: to test paths for this toolbox, consider resetting Matlab
% paths to default (i.e., using restoredefaultpath) then running this code
% block.
GitPath = 'F:\InnovationLibrary'; % <-- Adjust this for particular machine
addpath(genpath([GitPath '\' 'Toolboxes\MAP Filter']))

%Set images path
raw_path = "F:\GridRemoving\RAW_files\";
tifs_path = "F:\GridRemoving\tiff_files\";
images_raw = raw_path + "Images\";
maps_raw = raw_path + "Maps\";
images_tifs = fullfile(tifs_path , "Images\");
maps_tifs = tifs_path + "Maps\";
%get file names
images_raw_files = dir(images_raw + "*.raw");
file_names = string.empty;
for i = 1:length(images_raw_files)
    name_raw = images_raw_files(i).name;
    name_root = strsplit(string(name_raw) , '.');
    name_root = name_root(1);
    file_names = [file_names, name_root];
end
%% 
% raw_file_header = struct('Rows',{'3268'},'Columns',{'2756'},'Type',{'uint16'},'SkipBits',{'0'},'BitOrder',{'n'});
% temp = inputdlg( {'DataType, see "fread" documentation for DataType options',...
%         'Rows','Columns','SkipBits','BitOrder'}, 'Raw File Header', 1, ...
%         {raw_file_header.Type, raw_file_header.Rows, raw_file_header.Columns, raw_file_header.SkipBits, raw_file_header.BitOrder}, 'on' );raw_file_header(1).Type = temp{1};
% raw_file_header(1).Rows = str2double( temp{2} );
% raw_file_header(1).Columns = str2double( temp{3} );
% raw_file_header(1).SkipBits = str2double( temp{4} );
% raw_file_header(1).BitOrder = temp{5};
%% 

%index = 15;
StripConfig.UserThresh = 1;  % Preset(0) or user determined (1) threshold
StripConfig.PeakThreshold = 0.067; % Manual threshold (if used)

% Remove outliers
Stretch = [0.01, 0.99];

% Configure/generate spectral mask (suggested settings)

StripConfig.NotchType = 'Dolph';  % 'Gaussian', 'Dolph', 'Astroid'
StripConfig.AstroBlurAmountSize = 30; % Radius of Astro blur
StripConfig.AstroBlurAmountSigma = 10; % Sigma of Astro blur
StripConfig.FilterMode = 1; % Filter mode: 1 = circular only 2 = tubular only; 3 = tubular and circular
StripConfig.TubeWidth = 40;  %  tube width

StripConfig.CircWidth = 1500; % Notch window width

StripConfig.TubeLenFrac = 300; % tube length
StripConfig.TubeProtWidth = 2*StripConfig.TubeWidth; % Tube protective region
StripConfig.CircularProtWidth = 1; % Circular protective region
StripConfig.AstroScale = 2; % Scaling of astoid relative to Dolph window: 2 is a useful value for approx. matching of attenutation
StripConfig.nRings = 6; % Number of radial sets of circular filters

StripConfig.PeakGroupingBound = 10;  % PeakGroupingBound; used for peak location
StripConfig.PeakBlurSigma = 10; % PeakBlurSigma; used to blur the peak detection image
StripConfig.GausSigma = 20; % Sigma relating to Gaussian window (otherwise not used)
StripConfig.DCScaling = 1; % Scaling of circular windows in DC region (disabled for the value 1.0) 
StripConfig.TubeVHRatio = 1.5; % Set ratio of vertical to horizontal tube length (default = 1.0)


%figure; imagesc(FilterMask); 



%% 
% loop through all raw files and remove grid
for file = file_names
%file = file_names(index);
%file = "right_thigh_80kv_wed26"
raw_name = file + '.raw';
tif_name = file + '.tif';
map_name = file + '_map.raw';
fileIO = fopen(fullfile(images_raw , raw_name));
%fileIO = fopen("F:\GridRemoving\RAW_files\Images\arm_friday_21_october.raw");
if fileIO == -1
    errordlg('Check the Image path (normally typos cause this error)')
end
mapIO = fopen(fullfile(maps_raw , map_name));
%mapIO = fopen("F:\GridRemoving\RAW_files\Maps\arm_friday_21_october_map.raw");
if mapIO == -1
    errordlg('Check the Image path (normally typos cause this error)')
end
image = fread( fileIO, Inf, 'uint16', 0, 'n');
image = reshape( image, 3268, 2756, []);
image_nooutliers = remove_outliers_sort(image, Stretch);
image_nooutliers = image_nooutliers / max(max(image_nooutliers));

map = fread( mapIO, Inf, 'uint16', 0, 'n');
map = reshape( map, 3268, 2756, []);
map_nooutliers = remove_outliers_sort(map, Stretch);
map_nooutliers = map_nooutliers / max(max(map_nooutliers));

Guide = map_nooutliers;
%% Filter I image
% Generate mask
[FilterMask, FilterMaskTypes, FilterAdditional] =  MAPStrip_GenerateMaskWrapper_Wrap(StripConfig, Guide);
ToFilter = single(image_nooutliers);
StripConfig.FilterPadType = 3;
[FilteredTx, mask, TxSpectrumMasked, TxSpectrum] =  MAPStrip_ApplyMask_Lib(ToFilter, FilterMask, StripConfig.FilterPadType);

%imagesc_multiple({ToFilter FilteredTx},{'source','filtered'},{'colorbar, colormap(gray)'},1)

%save to tiff
FA_WriteTif_Lib(FilteredTx, char(fullfile(tifs_path, tif_name)));
fclose( fileIO );
fclose( mapIO );
    
end




function [x, thresh] = remove_outliers_sort(x, p, trim_top)
% p: vector of quantiles e.g. [0.05, 0.95] - values should be between 0 and
% 1
if (length(p)>=2)
    p = sort(p(1:2));
    thresh = quant_sort(x,p);
    x(x<thresh(1)) = thresh(1);
    x(x>thresh(2)) = thresh(2);
end
if (length(p) == 1)
    thresh = quant_sort(x,p);
    if (trim_top == 1)
        x(x<thresh) = thresh(1);
    else
        x(x>thresh) = thresh(1);
    end
end
end

function q = quant_sort(x,p)
% p: vector of quantiles e.g. [0.05, 0.95] - values should be between 0 and
% 1
p=p(:);
p(p<0)=0; % remove negative values
p(p>1)=1; % remove values greater than 1
%Remove NaNs
s = isnan(x);
if sum(s(:))>0
    x(s)=[];
    sprintf('%d Nan(s) removed', sum(s(:)))
end
% Remove Infs
s = isinf(x);
if sum(s(:)) > 0
    x(s)=[];
    sprintf('%d Inf(s) removed', sum(s(:)))
end
x=sort(x(:)); % sort vector/matrix (can this be improved?)
N=length(x);
%Find edges
x=[x(1);x;x(N)]';
%Find indices for each pixel
ind_x=[0 (((0.5:(N-0.5)))/N) 1];
% Interpolate indices
q=interp1(ind_x,x,p);
end