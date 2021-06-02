function foo
    input Integer x;
    output Integer[:,:] y;

algorithm
    y := identity(x);
end foo;
