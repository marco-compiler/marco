function foo
    input Real[:] x;
    input Real[:] y;
    output Real[:] z;

algorithm
    z := x - y;
end foo;
