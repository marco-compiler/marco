function foo
    input Integer[:] x;
    output Integer y;

    algorithm
        y := x[1] + x[2] + x[3];
end foo;
