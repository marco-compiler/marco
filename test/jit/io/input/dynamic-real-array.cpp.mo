function foo
    input Real[:] x;
    output Real y;

    algorithm
        y := x[1] + x[2] + x[3];
end foo;
