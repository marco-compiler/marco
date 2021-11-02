model ImplicitKepler
	Real[2] x(start = 3.7);
equation
	for i in 1:2 loop
		5.0 = x[i] - 2.72 * sin(x[i]);
	end for;
end ImplicitKepler;
