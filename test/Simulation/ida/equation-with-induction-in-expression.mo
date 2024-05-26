// RUN: marco %s --omc-bypass --model=InductionUsage --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","x[5]"
// CHECK: 0.0000,0.5000,0.5000,0.5000,0.5000,0.5000
// CHECK: 1.0000,0.7000,1.3000,1.7000,2.1000,2.5000

model InductionUsage
	Real[5] x(each start = 0.5, fixed = true);
equation
	5.0 * der(x[1]) = 1.0;

	for i in 2:5 loop
		5.0 * der(x[i]) = 2.0 * i;
	end for;
end InductionUsage;
