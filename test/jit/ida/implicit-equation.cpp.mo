model ImplicitEq
	Real x(start = 1.5);
equation
	x + x * x * x = 7.0;
end ImplicitEq;
