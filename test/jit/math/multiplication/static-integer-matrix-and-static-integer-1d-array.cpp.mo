function foo
    input Integer[4,3] x;
    input Integer[3] y;
    output Integer[4] z;

algorithm
    z := x * y;
end foo;
