function foo
    input Integer[:,:] x;
    output Integer[:,:] y;

algorithm
    y := symmetric(x);
end foo;
