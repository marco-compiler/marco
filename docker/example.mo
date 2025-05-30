function externalLog
    input Integer b;
    input Integer n;
    output Integer ris;
    external "C"
        ris=discreteLog(b,n)
        annotation(
            Library = "myLib",
            SourceDirectory="lib.c"

        );
end externalLog;

model SimpleFirstOrder
    Real x(start = 0, fixed = true);
equation
    der(x) = 1 - externalLog(2, x);
end SimpleFirstOrder;