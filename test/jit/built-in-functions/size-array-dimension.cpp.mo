function foo
    input Integer[:,:] x;
    output Integer y;

algorithm
    y := size(x, 2);
end foo;
