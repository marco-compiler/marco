model IdaInduction
	Real[6] x;
	Real[3] y;
equation
	for i in 1:6 loop
		x[i] = i + sin(time * 100);
	end for;
	for i in 1:3 loop
		der(y[i]) = x[i+3];
	end for;
end IdaInduction;
