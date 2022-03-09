function f1
    input Integer x;
    output Integer y;

algorithm
    y := -x;
end f1;

function foo
    input Integer[3] x;
    output Integer[3] y;

algorithm
    y := f1(x);
end foo;
