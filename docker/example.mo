function externalLogReal
  input Integer b;
  input Integer n;
  output Real ris; 
external "C"
  ris = discreteLog(b, n);
end externalLogReal;

model SimpleFirstOrder
  Real x;
equation
  x = externalLogReal(2, 100);
end SimpleFirstOrder;