function foo
    output Integer y;

algorithm
    y := 0;

    for i in 1:10 loop
        if y == 0 then
            y := 1;
            break;
        end if;

        y := 0;
    end for;
end foo;
