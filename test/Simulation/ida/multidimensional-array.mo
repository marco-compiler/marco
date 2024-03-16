// RUN: marco --omc-bypass --model=MultidimensionalArray --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1,1]","x[1,2]","x[1,3]","x[2,1]","x[2,2]","x[2,3]"
// CHECK: 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
// CHECK: 1.0000,1.0000,2.0000,2.0000,1.0000,2.0000,2.0000

model MultidimensionalArray
	Real[2, 3] x(each start = 0, fixed = true);
equation
	for i in 1:2 loop
		der(x[i, 1]) = 1.0;

		for j in 2:3 loop
			der(x[i, j]) = 2.0;
		end for;
	end for;
end MultidimensionalArray;
