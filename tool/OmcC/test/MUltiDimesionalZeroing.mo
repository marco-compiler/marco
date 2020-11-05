class MultiDimZeroingExample
	Real[10, 10] v1;
	Real[10, 10] v2;
equation
	for j in 0:9 loop
		for i in 0:9 loop
			v1[j+1, i+1] = v2[j+1, i+1];
		end for;
	end for;
	
	for j in 7:11 loop
		for i in 0:9 loop
			v2[j-1, i+1] = -v1[j-1, i+1];
		end for;
	end for;

	for j in 2:6 loop
		for i in 0:9 loop
			v2[j-1, i+1] = -v1[j-1, i+1];
		end for;
	end for;

end MultiDimZeroingExample;
