model ExplicitEq
	Real[2] x;
equation
	for i in 1:2 loop
		3 = x[i] + i;
	end for;
end ExplicitEq;
