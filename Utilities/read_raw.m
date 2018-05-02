filepath ="X:\Raw_Data\Critical_Raw_Data\All NSTC Data\NSTC 05-06-17\ProcessedData\Knee\Knee 75kV No MAP ASG 6p4mAs 105cm\B_ (1).raw"


raw_file_header(1).Type = temp{1};
raw_file_header(1).Rows = str2double( temp{2} );
raw_file_header(1).Columns = str2double( temp{3} );
raw_file_header(1).SkipBits = str2double( temp{4} );
raw_file_header(1).BitOrder = temp{5};
    
fileIO = fopen(filepath);

if fileIO == -1
    errordlg('Check the Image path (normally typos cause this error)')
end
image = fread( fileIO, Inf, raw_file_header.Type, raw_file_header.SkipBits, raw_file_header.BitOrder);
image = reshape( image, raw_file_header.Rows, raw_file_header.Columns, []);
fclose( fileIO );
imtool(image)