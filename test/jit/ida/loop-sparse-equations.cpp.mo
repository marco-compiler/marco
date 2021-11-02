model SparseLoop
	Real[3] x;
equation
	x[1] + x[2] = 3;
	x[1] + x[3] = 2;
	x[2] + x[3] = 1;
end SparseLoop;
