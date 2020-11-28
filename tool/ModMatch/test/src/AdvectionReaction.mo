model AdvectionReaction
  parameter Real mu = 1000;
  constant Real alpha = 0.5;
  parameter Real u_in = 1;
  Real[10] u(start = 0.0);
equation
  der(u[1]) = ((-u[1]) + 1)*10 - mu*u[1]*(u[1] - alpha)*(u[1] - 1);
  for j in 2:10 loop
	der(u[j]) = ((-u[j]) + u[j-1])*10 - mu*u[j]*(u[j] - alpha)*(u[j] - 1);
  end for;
end AdvectionReaction;

 
