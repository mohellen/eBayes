function mat = read_file(dat_file)

    f = fopen(dat_file,'r'); % open file for reading
    mat = []; % initialize output

    tline = fgets(f);  % get a line
    while ischar(tline),
        cell = textscan(tline, '%f');   % read line
        row = [cell{1}]';   % unpack data from cell
        mat = [mat ; row];  % append to output matrix
        tline = fgets(f); % get next line
    end

    fclose(f);
    return;
end