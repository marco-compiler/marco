function foo
    input Integer x;
    output Integer y;

algorithm
    y := 0;

    for i in 1:x loop
        y := y + i;
    end for;
end foo;
