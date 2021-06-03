function foo
    input Integer[4] x;
    input Integer[4,3] y;
    output Integer[3] z;

algorithm
    z := x * y;
end foo;
