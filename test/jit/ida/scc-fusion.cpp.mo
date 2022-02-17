model SccFusion
	Real[3] x;
	Real[2] y;
equation
	for i in 1:2 loop
		x[i] = 2.0 * y[i];
	end for;
	for i in 3:3 loop
		x[i] = 2.0;
	end for;
	for i in 1:1 loop
		x[i + 1] = 3.0 * y[i];
	end for;
	for i in 2:2 loop
		4.0 = 3.0 * y[i];
	end for;
end SccFusion;
