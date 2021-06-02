function foo
    input Integer[:,:] x;
    output Integer[:,:] y;

algorithm
    y := transpose(x);
end foo;
