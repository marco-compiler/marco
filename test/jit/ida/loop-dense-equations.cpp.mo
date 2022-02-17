model DenseLoop
	Real[2] x;
	Real[2] y;
	Real[2] z;
equation
	for i in 1:2 loop
		x[i] + y[i] - z[i] = 1.0;
		x[i] - y[i] + z[i] = 2.0;
		-x[i] + y[i] + z[i] = 3.0;
	end for;
end DenseLoop;
