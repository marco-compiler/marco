function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;

algorithm
    y := linspace(start, stop, n);
end foo;
