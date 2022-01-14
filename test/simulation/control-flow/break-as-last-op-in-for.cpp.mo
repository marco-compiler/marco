function foo
    output Integer y;

algorithm
    y := 0;

    for i in 1:10 loop
        y := 1;
        break;
    end for;
end foo;
