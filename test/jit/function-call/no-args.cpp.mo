function f1
    output Integer x;

algorithm
    x := 1;
end f1;

function foo
    output Integer x;

algorithm
    x := f1();
end foo;
