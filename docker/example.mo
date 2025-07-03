  function externalLog
    input Integer b;
    input Integer n;
    output Integer ris;
  external "C"
    ris = discreteLog(b,n);
  end externalLog;

  model SimpleFirstOrder
    Real x(start = 0, fixed = true);
  equation
    der(x) = 4-externalLog(2,256);
  end SimpleFirstOrder;
  