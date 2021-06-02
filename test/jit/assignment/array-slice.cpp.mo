function foo
    input Integer[2] x;
    input Integer[2] y;
    input Integer[2] z;
    output Integer[3,2] t;

algorithm
    t[1] := x;
    t[2] := y;
    t[3] := z;
end foo;
