model MultidimDer
	Real[2, 3] x(start = 0.0);
equation
	for i in 1:2 loop
		der(x[i, 1]) = 1.0;
		for j in 2:3 loop
			der(x[i, j]) = 2.0;
		end for;
	end for;
end MultidimDer;
