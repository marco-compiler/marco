class ZeroingExample
	Real[10] v1;
	Real[10] v2;
equation
	for j in 1:10 loop
		v1[j] = v2[j];
	end for;
	
	for j in 2:6 loop
		v2[j-1]	 = -v1[j-1];
	end for;

	for j in 7:11 loop
		v2[j-1]	 = -v1[j-1];
	end for;

end ZeroingExample;

