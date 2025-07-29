function externalLogReal
  input Integer b;
  input Integer n;
  output Real ris; 
external "C"
  discreteLog(b, n, ris);
end externalLogReal;

model SimpleFirstOrder
  Real x;
equation
  x = externalLogReal(2, 100);
end SimpleFirstOrder;