function externalLog
  input Integer b;
  input Real n;
  output Integer ris;
  external "C" ris = discreteLog(b, n)
    annotation(
      Include = "#include \"lib.c\"",
      Library = "myLib"
    );
end externalLog;


model SimpleFirstOrder
  Real x(start = 0, fixed = true);
equation
  der(x) = 1 - externalLog(2, x);
end SimpleFirstOrder;
