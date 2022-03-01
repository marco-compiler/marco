model ArraysWithState
	Real[3] x;
	Real[3] y;
equation
	for i in 1:3 loop
		der(x[i]) = 1.0;
		der(y[i]) = 3 + x[i];
	end for;
end ArraysWithState;
