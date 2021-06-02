function foo
    input Integer x;
    output Integer y;

protected
    Integer[2] z;

algorithm
    z[0] := x * 2;
    z[1] := z[0] + 1;
    y := z[1];
end foo;
