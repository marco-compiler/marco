function foo
    output Integer y;

algorithm
    y := 0;

    while true loop
        y := 1;
        break;
    end while;
end foo;
