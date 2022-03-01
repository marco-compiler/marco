model Loop8
    Real[4] x;
    Real y;
equation
    for i in 1:2 loop
        x[i] + x[i+2] = 3;
    end for;
    x[3] = y + 7;
    for i in 3:4 loop
        x[i] - x[i-2] = 1;
    end for;
end Loop8;
