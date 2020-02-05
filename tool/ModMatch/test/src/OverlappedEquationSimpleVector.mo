package SimpleVector
	Real[10] vector(start = 10.0);
	equation
	for i in 1:5 loop
		vector[i] = 10;
	end for;
	for i in 4:7 loop
		vector[i] = 10;
	end for;
end SimpleVector;
