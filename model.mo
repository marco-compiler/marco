model SimpleDer 
  final parameter Real tau = 5.0;
  Real[4] x1(start = 0.0);
  Real y(start=9.99);
equation
  tau*der(x1[1]) = 1.0;
  for i in 2:4 loop
	tau*der(x1[i]) = 2.0*i;
  end for;
end SimpleDer;