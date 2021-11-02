model Sched4
	Real[3] x;
	Real[2] y;
	Real[4] w;
	Real z;
equation
	for i in 3:3 loop
		w[i] - w[i+1] = y[2] - w[i-2];
	end for;
	for j in 1:2 loop
		y[j] = w[j] + x[j];
	end for;
	for j in 1:1 loop
		w[j] + w[j+1] = x[3];
	end for;
	w[4] = 7 - w[3];
	w[3] = z - y[1];
	for i in 1:3 loop
		x[i] = 5;
	end for;
	w[1] - w[2] = 3;
end Sched4;
