function foo
    input Integer[2,3] x;
    output Integer y;
    output Integer z;

    algorithm
        y := x[1,1] + x[1,2] + x[1,3];
        z := x[2,1] + x[2,2] + x[2,3];
end foo;
