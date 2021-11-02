model SimpleDerSub
	Real y;
	Real x;
equation
	der(x) = y;
	y = 2.0;
end SimpleDerSub;
