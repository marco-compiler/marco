function foo
    input Integer[:] x;
    output Integer[:] y;

algorithm
    y := x;
    y[1] := 10;
end foo;
