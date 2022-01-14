function foo
    input Boolean[4] x;
    input Boolean[4] y;
    output Boolean[4] z;

algorithm
    z := x or y;
end foo;
