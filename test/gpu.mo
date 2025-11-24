model M
    Real[10,20] x;
equation
    for i in 1:10 loop
        for j in 1:20 loop
            x[i,j] = 1;
        end for;
    end for;
end M;
