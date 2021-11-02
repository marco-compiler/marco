model Robertson
	Real y1(start = 1.0);
	Real y2(start = 0.0);
	Real y3(start = 0.0);
equation
	der(y1) = -0.04 * y1 + 1e4 * y2 * y3;
	der(y2) = +0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2;
	0 = y1 + y2 + y3 - 1;
end Robertson;
