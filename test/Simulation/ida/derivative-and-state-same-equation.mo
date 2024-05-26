// RUN: marco %s --omc-bypass --model=ArraysWithState --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","y[1]","y[2]","y[3]"
// CHECK: 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
// CHECK: 1.0000,1.0000,1.0000,1.0000,3.5000,3.5000,3.5000

model ArraysWithState
	Real[3] x(each start = 0, fixed = true);
	Real[3] y(each start = 0, fixed = true);
equation
	for i in 1:3 loop
		der(x[i]) = 1.0;
		der(y[i]) = 3 + x[i];
	end for;
end ArraysWithState;
