// RUN: marco %s --omc-bypass --model=ImplicitKepler --solver=ida -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]"
// CHECK: 0.0000,3.6577,3.6577
// CHECK: 1.0000,3.6577,3.6577

model ImplicitKepler
	Real[2] x(each start = 3.6, fixed = false);
equation
	for i in 1:2 loop
		5.0 = x[i] - 2.72 * sin(x[i]);
	end for;
end ImplicitKepler;
