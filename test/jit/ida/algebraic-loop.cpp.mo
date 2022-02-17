model AlgebraicLoop
	Real[5] x;
equation
	x[1] + x[2] = 2;
	x[1] - x[2] = 4;
	x[3] + x[4] = 1;
	x[3] - x[4] = -1;
	x[5] = x[4] + x[1];
end AlgebraicLoop;
