function foo
    input Integer[2,2] x;
    input Integer y;
    output Integer[2,2] z;

algorithm
    z := x ^ y;
end foo;
