function cp_T
	input Real T;
	output Real cp;
	protected Real[2] u;
	protected Real x;
algorithm
	u := {4.5, 3.3};
	x := 5.2;
	for i in 1:2 loop
		x := x + {2.7, 10.9}[i] * u[i];
	end for;
	cp := x * T;
end cp_T;

function pder_cp_t = der(cp_T, T);

model CallOp6
	Real z;
	Real w;
equation
	der(w) = cp_T(3.0);
	der(z) = pder_cp_t(3.0);
end CallOp6;
