// RUN: marco %s --omc-bypass --model=ImplicitEquation --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x"
// CHECK: 0.0000,1.7392
// CHECK: 1.0000,1.7392

model ImplicitEquation
	Real x(start = 0, fixed = false);
equation
	x + x * x * x = 7.0;
end ImplicitEquation;
