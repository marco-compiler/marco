model ArrayInit
	Real[6] x = {1, 2.0, 3, 4, 5, 6};
	Real[6] y = {3, 4, 3.0, 4, 3, 4};
equation
	for i in 1:6 loop
		x[i] = 1;
	end for;
	for j in 1:3 loop
		der(y[j]) = x[j+2] + 2;
		der(y[j+3]) = x[j+1] - 2;
	end for;
end ArrayInit;
