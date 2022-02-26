model UnevenRow
	Real[3, 4] x;
equation
	for i in 1:3 loop
		for j in 1:4 loop
			der(x[i, j]) = 2 * der(x[2, 2]) - 4;
		end for;
	end for;
end UnevenRow;
