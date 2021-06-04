function f1
    output Integer[3] y;

algorithm
    y[1] := 1;
    y[2] := 2;
    y[3] := 3;
end f1;

function foo
    output Integer[3] y;

algorithm
    y := f1();
end foo;
