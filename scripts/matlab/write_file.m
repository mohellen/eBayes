function write_file(mat, dat_file)

    f = fopen(dat_file,'w'); % open file for writing
    
    rows = size(mat,1);
    cols = size(mat,2);
    
    for i=1:rows,
        for j=1:cols,
            fprintf(f,'%.6f ', mat(i,j));
        end
        fprintf(f,'\n');
    end
    return
end