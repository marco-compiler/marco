function foo
    input Integer n1;
    input Integer n2;
    output Integer[:,:] y;

algorithm
    y := zeros(n1, n2);
end foo;
