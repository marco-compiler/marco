function foo
    input Integer x;
    output Integer y;

algorithm
    if x == 0 then
        y := 1;
    elseif x > 0 then
        y := 2;
    else
        y := 3;
    end if;
end foo;
