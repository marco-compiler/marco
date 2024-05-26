// RUN: marco %s --omc-bypass --model=InductionInExpression --solver=euler-forward -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]"
// CHECK: 0.000000,1.000000,2.000000,3.000000
// CHECK: 1.000000,2.000000,3.000000,4.000000

model InductionInExpression
	Real[3] x;
equation
	for i in 1:3 loop
		x[i] = i + time;
	end for;
end InductionInExpression;
