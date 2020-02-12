model SimpleAdvection 
  parameter Real L = 10;
  final parameter Real l = L/4;
  Real[1] u(start = 1.0);
  parameter Real Tin = 300;
  Real[4] T(start = 0.0);
  Real[3] Ttilde(start = 300.0);
  parameter Real Tout = 0.0;
equation
  for j in 1:3 loop
	der(Ttilde[j]) = u/l*(T[j]-T[j+1]);
  end for;
  T[1] = Tin;
  T[4] = Tout;
  for j in 1:3 loop
    Ttilde[j] = T[j+1];
  end for;
end SimpleAdvection;


