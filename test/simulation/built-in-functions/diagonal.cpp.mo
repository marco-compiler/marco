function foo
    input Integer[:] x;
    output Integer[:,:] y;

algorithm
    y := diagonal(x);
end foo;
