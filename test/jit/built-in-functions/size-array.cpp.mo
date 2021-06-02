function foo
    input Integer[:,:] x;
    output Integer[2] y;

algorithm
    y := size(x);
end foo;
