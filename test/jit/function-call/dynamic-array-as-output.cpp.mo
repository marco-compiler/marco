function f1
    input Real x;
    input Integer n;
    output Real[n] y;

algorithm
    for i in 1:n loop
        y[i] := x * i;
    end for;
end f1;

function foo
    output Real[3] y;

algorithm
    y := f1(2, 3);
end foo;
