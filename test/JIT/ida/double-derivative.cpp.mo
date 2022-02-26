model DoubleDer
	Real x;
	Real y;
equation
	der(x) = 2.0;
	der(y) = time;
end DoubleDer;
