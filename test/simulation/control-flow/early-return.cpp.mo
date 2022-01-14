function foo
    output Integer y;

algorithm
    y := 1;

    if true then
        return;
    end if;

    y := 0;
end foo;
