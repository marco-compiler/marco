function externalLogReal
  input Integer b;
  input Integer n;
  output Real ris; 
external "C"
  ris = discreteLog(b, n);
end externalLogReal;

model SimpleFirstOrder
  Real x(start = 0, fixed = true);
equation
  der(x) = 10 - externalLogReal(2, 256);
end SimpleFirstOrder;