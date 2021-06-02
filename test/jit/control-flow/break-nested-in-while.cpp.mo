function foo
    output Integer y;

algorithm
    y := 0;

    while true loop
        if y == 0 then
            y := 1;
            break;
        end if;

        y := 0;
    end while;
end foo;
