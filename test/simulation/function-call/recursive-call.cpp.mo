function foo
    input Real[3] x;
    input Integer i;
    output Real y;

algorithm
    y := 0;

    if i <= 3 then
        y := x[i] + foo(x, i + 1);
    end if;
end foo;
