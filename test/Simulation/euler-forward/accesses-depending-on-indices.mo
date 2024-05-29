// RUN: marco %s --omc-bypass --model=AccessesDependingOnIndices --solver=euler-forward -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1,1]","x[1,2]","x[1,3]","x[1,4]","x[2,1]","x[2,2]","x[2,3]","x[2,4]","x[3,1]","x[3,2]","x[3,3]","x[3,4]"
// CHECK: 0.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000
// CHECK: 1.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000,4.000000

model AccessesDependingOnIndices
	Real[3, 4] x;
equation
	for i in 1:3 loop
		for j in 1:4 loop
			x[i, j] = 2 * x[2, 2] - 4;
		end for;
	end for;
end AccessesDependingOnIndices;
