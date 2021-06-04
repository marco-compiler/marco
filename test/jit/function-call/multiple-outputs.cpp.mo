function f1
    input Integer x;
    output Integer y;
    output Integer z;

algorithm
    y := 2 * x;
    z := 3 * x;
end f1;

function foo
    input Integer x;
    output Integer y;
    output Integer z;

algorithm
    (y, z) := f1(x);
end foo;
