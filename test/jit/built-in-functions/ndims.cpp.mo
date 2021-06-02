function foo
    input Integer[3, 2] x;
    output Integer y;

algorithm
    y := ndims(x);
end foo;
