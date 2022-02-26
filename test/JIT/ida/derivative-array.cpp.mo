model DerArray
	final parameter Real tau = 5.0;
	Real[5] x(start = 0.5);
equation
	tau*der(x[1]) = 1.0;
	for i in 2:5 loop
		tau*der(x[i]) = 2.0 * i;
	end for;
end DerArray;
