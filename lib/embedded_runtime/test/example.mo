model SimpleDer 
  final parameter Real tau = 5.0;
  Real[10] x(start = 0.0);
equation
  tau*der(x[1]) = 1.0;
  for i in 2:10 loop
  tau*der(x[i]) = 2.0*i;
  end for;
end SimpleDer;
