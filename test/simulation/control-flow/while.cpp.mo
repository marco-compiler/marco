function foo
    input Integer x;
    output Integer y;

protected
    Integer i;

algorithm
    y := 1;
    i := 0;

    while i < x loop
        y := y + x;
        i := i + 1;
    end while;
end foo;
