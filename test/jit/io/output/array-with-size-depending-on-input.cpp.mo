function foo
    input Integer x;
    output Integer[x + x] y;

    algorithm
        for i in 1 : (x + x) loop
            y[i] := i;
        end for;
end foo;
