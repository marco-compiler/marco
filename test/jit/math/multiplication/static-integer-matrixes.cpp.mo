function foo
    input Integer[2,3] x;
    input Integer[3,2] y;
    output Integer[2,2] z;

algorithm
    z := x * y;
end foo;
